# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from Model.Model import Model
from Orchestrator.Orchestrator import Orchestrator
from utils import add_dependencies, allreduce, receive, send
from chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)


class DualPipe(Orchestrator):
    def __init__(self, 
        model: Model,
        config):
        self.model = model
        self.dp_size = config["parallelism"]["dp_size"]
        self.pp_size = config["parallelism"]["pp_size"]
        self.tp_size = config["parallelism"]["tp_size"]
        self.num_npus = self.dp_size * self.pp_size * self.tp_size
        self.num_microbatches = config["model"]["num_microbatches"]
        self.num_microbatches_per_half = self.num_microbatches // 2
        self.scale = config["model"]["scale"]

    def generate_comm_groups(self):
        comm_groups = defaultdict(list)

        # generate comm groups for data parallel groups        
        for dp_group in range(self.dp_size):
            for pp_stage in range(self.pp_size):
                    # dp all-reduce group consists of all ranks that share the same pipeline stage and tensor parallel shard
                    npu_id = dp_group * self.pp_size + pp_stage
                    pp_name = "" if self.pp_size <= 1 else f"pp_{pp_stage}"
                    comm_groups[f"{pp_name}"].append(npu_id)

        return comm_groups

    def exec(self) -> dict:
        num_params = self.model.num_params
        B = self.model.get_batch_size()
        S = self.model.get_sequence_len()
        d = self.model.get_hidden_size()
        b = self.model.get_bytes_per_val()

        layers_per_pipeline_stage = self.model.get_num_layers() // self.pp_size
        #print(f"Layers per pipeline stage: {layers_per_pipeline_stage}")
        print(B, S, d, b, self.scale, self.num_microbatches)
        print("calculating pp_comm_size = (B*S*d*b * scale) / num_microbatches where B={B}, S={S}, d={d}, b={b}, scale={self.scale}, num_microbatches={self.num_microbatches}")
        pp_comm_size = int((B*S*d*b * self.scale) / self.num_microbatches)
        print(pp_comm_size)
        dp_comm_size = int(self.scale * num_params * b / self.tp_size / self.pp_size)

        # ##print(f"Num params: {num_params:,.2f}")
        # ##print(f"Pipeline comm size: {pp_comm_size / 1024 / 1024:,.2f} MB")
        # ##print(f"DP comm size: {dp_comm_size / 1024 / 1024 / 1024:,.2f} GB")

        nodes = defaultdict(list)
        forward_nodes = defaultdict(list)
        backward_nodes = defaultdict(list)
        for dp_group in range(self.dp_size):
            ##print(f"================ DP GROUP {dp_group} ================")
            for pp_stage in range(self.pp_size):
                    npu_id = dp_group * self.pp_size + pp_stage   # tp_shard always 0
                    print(f"Processing NPU ID: {npu_id}")
                    nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
                    
                    # ##print(f"NPU {npu_id} - dp group: {dp_group}, pp stage: {pp_stage}, tp shard: {tp_shard}")
                    # -------------
                    # Forward pass
                    # -------------
                    prev_rcv = None
                    prev_comp = None
                    for b in range(self.num_microbatches_per_half):
                        ##print(f"--- Micro-batch {b} ---")
                        rcv_node = None
                        if pp_stage != 0 and self.pp_size > 1:
                            print(f"RCV ({npu_id - 1} -> {npu_id}) for micro_batch {b}")
                            rcv_node = receive(npu_id - 1, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_FWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(rcv_node)
                            prev_rcv = rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            ##print(f"Layer {layer} in pipeline stage {pp_stage}")
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_micro_batch{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"0")
                            #print(f"  Compute nodes: {[node.name for node in cmp_nodes]}")
                            if layer == 0:
                                # print(rcv_node)
                                add_dependencies(cmp_nodes[0], [rcv_node, prev_comp])
                                print(f"Adding dependencies to {cmp_nodes[0].name} : {[rcv_node.name if rcv_node else None, prev_comp.name if prev_comp else None]}")
                            else:
                                add_dependencies(cmp_nodes[0], [prev_comp])
                                print(f"Adding dependency to {cmp_nodes[0].name} : {[prev_comp.name if prev_comp else None]}")
                            for node in cmp_nodes:
                                nodes[npu_id].append(node)
                                forward_nodes[npu_id].append(node)
                            prev_comp = cmp_nodes[-1]
                        
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            print(f"SND ({npu_id} -> {npu_id + 1}) for micro_batch {b}")
                            snd_node = send(npu_id, npu_id +1, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(snd_node)                
                    # -------------
                    # Backward pass
                    # -------------
                    print("Backward pass")
                    prev_comp = None
                    prev_rcv = None
                    for b in range(self.num_microbatches_per_half):
                        bck_rcv_node = None
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            print(f"RCV ({npu_id + 1} -> {npu_id}) for micro_batch {b}")
                            #print(prev_rcv)
                            bck_rcv_node = receive(npu_id + 1, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(bck_rcv_node)
                            prev_rcv = bck_rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_micro_batch{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"0")
                            if layer == 0:
                                print(f"Adding dependencies to {bck_cmp_nodes[0].name} : {[bck_rcv_node.name if bck_rcv_node else None, prev_comp.name if prev_comp else None]}")
                                # print(bck_rcv_node)
                                add_dependencies(bck_cmp_nodes[0], [bck_rcv_node, prev_comp])
                            else:
                                print(f"Adding dependency to {bck_cmp_nodes[0].name} : {[prev_comp.name if prev_comp else None]}")
                                add_dependencies(bck_cmp_nodes[0], [prev_comp])
                            for node in bck_cmp_nodes:
                                nodes[npu_id].append(node)
                                backward_nodes[npu_id].append(node)
                            prev_comp = bck_cmp_nodes[-1]
                        
                        
                        if pp_stage != 0 and self.pp_size > 1:
                            print(f"SND ({npu_id} -> {npu_id - 1}) for micro_batch {b}")
                            bck_snd_node = send(npu_id, npu_id - 1, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(bck_snd_node)


        print("-----")
        #print(forward_nodes[0][::4])
        print("Starting DualPipe scheduling")
        for j in range(self.num_npus):
            head_of_forward = forward_nodes[j][::(2*layers_per_pipeline_stage)]
            tail_of_forward= forward_nodes[j][1::(2*layers_per_pipeline_stage)]
            tail_of_backward = backward_nodes[j][1::(2*layers_per_pipeline_stage)]
            head_of_backward = backward_nodes[j][::(2*layers_per_pipeline_stage)]
            print(f"head_of_forward: {[node.name for node in head_of_forward]}")
            print(f"head_of_backward: {[node.name for node in head_of_backward]}")
            print(f"tail_of_backward: {[node.name for node in tail_of_backward]}")
            print(f"tail_of_forward: {[node.name for node in tail_of_forward]}")
            for i in range(len(head_of_forward)):
                if i-self.pp_size+j<0:
                    continue
                else:
                    
                    print(f"Adding dependency from {tail_of_backward[i - self.pp_size + j].name} to {head_of_forward[i].name}")
                    add_dependencies(head_of_forward[i], [tail_of_backward[i - self.pp_size+j]])
                        
            for i in range(len(head_of_backward)):
                if i + self.pp_size -j-1 < len(head_of_backward):
        
                    if(npu_id==7):
                        pass
                    print(f"Adding dependency from {tail_of_forward[i + self.pp_size - j -1].name} to {head_of_backward[i].name}")
                    add_dependencies(head_of_backward[i], [tail_of_forward[i + self.pp_size-j-1]])
    
        # pp_comm_size = int((B*S*d*b * self.scale) / self.num_microbatches)+1
        # print(pp_comm_size)
        # pp_comm_size = int((B*S*d*b * self.scale) / self.num_microbatches)
        # print(pp_comm_size)
        pp_comm_size+=1
        forward_nodes_of_first_half = forward_nodes
        backward_nodes_of_first_half = backward_nodes
    ### SYMMETRICAL PART BELOW ###
        print("Starting 2nd half of DualPipe scheduling")

        forward_nodes = defaultdict(list)
        backward_nodes = defaultdict(list)
        for dp_group in range(self.dp_size):
            ##print(f"================ DP GROUP {dp_group} ================")
            for pp_stage in range(self.pp_size):
                    
                    # print(f"--- Pipeline stage {pp_stage} ---")
                    npu_id = dp_group * self.pp_size + self.pp_size - pp_stage - 1    # tp_shard always 0
                    print(f"2nd time Processing NPU ID : {npu_id}")
                    # nodes[npu_id].append(GlobalMetadata(version="0.0.4"))
                
                    # -------------
                    # Forward pass
                    # -------------
                    print("Forward pass")
                    prev_rcv = None
                    prev_comp = None
                    for b in range(self.num_microbatches_per_half, self.num_microbatches):
                        
                        # print(f"--- Micro-batch {b} ---")
                        rcv_node = None
                        # print(f"pp_stage: {pp_stage}, pp_size: {self.pp_size}")
                        if pp_stage != 0 and self.pp_size > 1:
                            if npu_id-1 <= self.num_npus:
                                print(f"RCV ({npu_id+1} -> {npu_id}) for micro_batch {b}")
                                rcv_node = receive(npu_id+1, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_FWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                                nodes[npu_id].append(rcv_node)
                                prev_rcv = rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            ##print(f"Layer {layer} in pipeline stage {pp_stage}")
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            cmp_nodes = self.model.fwd(name=f"COMP_NODE_FWD_micro_batch{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"0")
                            #print(f"  Compute nodes: {[node.name for node in cmp_nodes]}")
                            if layer == 0:
                                add_dependencies(cmp_nodes[0], [rcv_node, prev_comp])
                            else:
                                add_dependencies(cmp_nodes[0], [prev_comp])
                            for node in cmp_nodes:
                                nodes[npu_id].append(node)
                                forward_nodes[npu_id].append(node)
                            prev_comp = cmp_nodes[-1]
                        
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            print(f"SND ({npu_id} -> {npu_id - 1}) for micro_batch {b}")
                            snd_node = send(npu_id, npu_id -1, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_FWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(snd_node)                
                    # -------------
                    # Backward pass
                    # -------------
                    print("Backward pass")
                    prev_comp = None
                    prev_rcv = None
                    for b in range(self.num_microbatches_per_half, self.num_microbatches):
                        # print(f"--- Micro-batch {b} ---")
                        bck_rcv_node = None
                        if pp_stage != self.pp_size - 1 and self.pp_size > 1:
                            print(f"RCV ({npu_id - 1} -> {npu_id}) for micro_batch {b}")
                            bck_rcv_node = receive(npu_id - 1, npu_id, pp_comm_size, parents=[prev_rcv], name=f"COMM_RECV_NODE_BCKWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(bck_rcv_node)
                            prev_rcv = bck_rcv_node
                        
                        for layer in range(layers_per_pipeline_stage):
                            current_layer = pp_stage * layers_per_pipeline_stage + layer
                            bck_cmp_nodes = self.model.bckwd(name=f"COMP_NODE_BCKWD_micro_batch{b}", npu_id=npu_id, layer=current_layer, num_batches=B/self.num_microbatches, pg_name=f"0")
                            if layer == 0:
                                add_dependencies(bck_cmp_nodes[0], [bck_rcv_node, prev_comp])
                            else:
                                add_dependencies(bck_cmp_nodes[0], [prev_comp])
                            for node in bck_cmp_nodes:
                                nodes[npu_id].append(node)
                                backward_nodes[npu_id].append(node)
                            prev_comp = bck_cmp_nodes[-1]
                        
                        
                        if pp_stage != 0 and self.pp_size > 1:
                            print(f"SND ({npu_id} -> {npu_id + 1}) for micro_batch {b}")
                            bck_snd_node = send(npu_id, npu_id + 1, pp_comm_size, parents=[prev_comp], name=f"COMM_SEND_NODE_BCKWD_micro_batch{b}_dp{dp_group}pp{pp_stage}")
                            nodes[npu_id].append(bck_snd_node)
        
        for j in range(self.num_npus): 
            head_of_forward = forward_nodes[j][::(2*layers_per_pipeline_stage)]
            tail_of_forward= forward_nodes[j][1::(2*layers_per_pipeline_stage)]
            tail_of_backward = backward_nodes[j][1::(2*layers_per_pipeline_stage)]
            head_of_backward = backward_nodes[j][::(2*layers_per_pipeline_stage)]
            # print(f"Processing NPU {j}")    
            for i in range(len(head_of_forward)):
                # print(i)
                if i-j-1<0 or i - j -1 >= len(tail_of_backward):
                    continue
                else:
                    add_dependencies(head_of_forward[i], [tail_of_backward[i - j -1]])
                    print(f"Added dependency from node {tail_of_backward[i - j -1  ].name} to node {head_of_forward[i].name} ")
                
            for i in range(len(head_of_backward)):
                if i + j < len(head_of_backward):
                    print(f"Added dependency from node {tail_of_forward[i +  j ].name} to node {head_of_backward[i].name} ")
                    add_dependencies(head_of_backward[i], [tail_of_forward[i + j]])
                    
                    

       
      
       
        print(B, S, d, b, self.scale, self.num_microbatches)
        print(pp_comm_size)

#         # target the heads (starts) and tails (ends) using your step-2 logic
#         first_half_heads = forward_nodes_of_first_half[3][::2]   # mb 0, 1, 2... 9
#         first_half_tails = forward_nodes_of_first_half[3][1::2]  # ends of mb 0, 1... 9

#         second_half_heads = forward_nodes[3][::2]                # mb 10, 11, 12... 19
#         second_half_tails = forward_nodes[3][1::2]               # ends of mb 10, 11... 19

# # --- Constraint 1: Microbatch 1-9 cannot start until 10-18 finishes ---
# # i=0 refers to mb 1 in the first half, depending on i=0 (mb 10) in the second half
#         for i in range(9):
#             head_node = first_half_heads[i + 1] # mb 1, 2, 3... 9
#             dep_node = second_half_tails[i]     # mb 10, 11, 12... 18
            
#             print(f"Added dependency: {head_node.name} (MB{i+1}) waits for {dep_node.name} (MB{i+10})")
#             add_dependencies(head_node, [dep_node])

#         # --- Constraint 2: Microbatch 11 cannot start until 0 finishes ---
#         # Index 1 of second_half is MB11. Index 0 of first_half is MB0.
#         mb11_head = second_half_heads[1]
#         mb0_tail = first_half_tails[0]

#         print(f"Added dependency: {mb11_head.name} (MB11) waits for {mb0_tail.name} (MB0)")
#         add_dependencies(mb11_head, [mb0_tail])
#         return nodes

# target the heads (starts) and tails (ends)
# heads: attention_compute | tails: ffwd_compute
    #     first_half_heads = forward_nodes_of_first_half[3][::2]   # mb 0, 1, 2... 9
    #     first_half_tails = forward_nodes_of_first_half[3][1::2]  # mb 0, 1, 2... 9

    #     second_half_heads = forward_nodes[3][::2]                # mb 10, 11, 12... 19
    #     second_half_tails = forward_nodes[3][1::2]               # mb 10, 11, 12... 19

    #     for i in range(10):
    #         # 1. Make MB(10+i) wait for MB(i)
    #         # Example: MB10 waits for MB0, MB11 waits for MB1...
    #         head_10_plus = second_half_heads[i]
    #         tail_i = first_half_tails[i]
    #         print(f"Zig: {head_10_plus.name} (MB{i+10}) waits for {tail_i.name} (MB{i})")
    #         add_dependencies(head_10_plus, [tail_i])

    #         # 2. Make MB(i+1) wait for MB(10+i)
    #         # Example: MB1 waits for MB10, MB2 waits for MB11...
    #         if i < 9:  # Avoid index out of bounds for MB9
    #             head_next_i = first_half_heads[i+1]
    #             tail_10_plus = second_half_tails[i]
    #             print(f"Zag: {head_next_i.name} (MB{i+1}) waits for {tail_10_plus.name} (MB{i+10})")
    #             add_dependencies(head_next_i, [tail_10_plus])
    #     # target the heads (starts) and tails (ends) for node index 4
    # # heads: first node (attention) | tails: second node (ffwd)
    #     first_half_heads = forward_nodes_of_first_half[4][::2]   # mb 0, 1, 2... 9
    #     first_half_tails = forward_nodes_of_first_half[4][1::2]  # mb 0, 1, 2... 9

    #     second_half_heads = forward_nodes[4][::2]                # mb 10, 11, 12... 19
    #     second_half_tails = forward_nodes[4][1::2]               # mb 10, 11, 12... 19

    #     for i in range(10):
    #         # 1. Make First Half wait for Second Half counterpart
    #         # Example: MB0 waits for MB10, MB1 waits for MB11...
    #         head_i = first_half_heads[i]
    #         tail_10_plus = second_half_tails[i]
    #         print(f"Reverse Zig: {head_i.name} (MB{i}) waits for {tail_10_plus.name} (MB{i+10})")
    #         add_dependencies(head_i, [tail_10_plus])

    #         # 2. Make Next Second Half wait for current First Half
    #         # Example: MB11 waits for MB0, MB12 waits for MB1...
    #         if i < 9:  # Avoid index out of bounds for the heads of 10-19
    #             head_next_10_plus = second_half_heads[i+1]
    #             tail_i = first_half_tails[i]
    #             print(f"Reverse Zag: {head_next_10_plus.name} (MB{i+11}) waits for {tail_i.name} (MB{i})")
    #             add_dependencies(head_next_10_plus, [tail_i])
    #     # heads: usually 'bwd_ffwd' or first node of bwd pair | tails: second node of bwd pair
    #     first_half_bwd_heads = backward_nodes_of_first_half[3][::2]   # mb 0, 1... 9
    #     first_half_bwd_tails = backward_nodes_of_first_half[3][1::2]  # mb 0, 1... 9

    #     second_half_bwd_heads = backward_nodes[3][::2]                # mb 10, 11... 19
    #     second_half_bwd_tails = backward_nodes[3][1::2]               # mb 10, 11... 19

    #     for i in range(10):
    #         # 1. First Half waits for Second Half counterpart (10->0, 11->1)
    #         head_i = first_half_bwd_heads[i]
    #         tail_10_plus = second_half_bwd_tails[i]
    #         print(f"NPU3 Bwd Zig: {head_i.name} (Bwd MB{i}) waits for {tail_10_plus.name} (Bwd MB{i+10})")
    #         add_dependencies(head_i, [tail_10_plus])

    #         # 2. Next Second Half waits for current First Half (0->11, 1->12)
    #         if i < 9:
    #             head_next_10_plus = second_half_bwd_heads[i+1]
    #             tail_i = first_half_bwd_tails[i]
    #             print(f"NPU3 Bwd Zag: {head_next_10_plus.name} (Bwd MB{i+11}) waits for {tail_i.name} (Bwd MB{i})")
    #             add_dependencies(head_next_10_plus, [tail_i])

    #     first_half_bwd_heads = backward_nodes_of_first_half[4][::2]   # mb 0, 1... 9
    #     first_half_bwd_tails = backward_nodes_of_first_half[4][1::2]  # mb 0, 1... 9

    #     second_half_bwd_heads = backward_nodes[4][::2]                # mb 10, 11... 19
    #     second_half_bwd_tails = backward_nodes[4][1::2]               # mb 10, 11... 19

    #     for i in range(10):
    #         # 1. Second Half waits for First Half counterpart (0->10, 1->11)
    #         head_10_plus = second_half_bwd_heads[i]
    #         tail_i = first_half_bwd_tails[i]
    #         print(f"NPU4 Bwd Zig: {head_10_plus.name} (Bwd MB{i+10}) waits for {tail_i.name} (Bwd MB{i})")
    #         add_dependencies(head_10_plus, [tail_i])

    #         # 2. Next First Half waits for current Second Half (10->1, 11->2)
    #         if i < 9:
    #             head_next_i = first_half_bwd_heads[i+1]
    #             tail_10_plus = second_half_bwd_tails[i]
    #             print(f"NPU4 Bwd Zag: {head_next_i.name} (Bwd MB{i+1}) waits for {tail_10_plus.name} (Bwd MB{i+10})")
    #             add_dependencies(head_next_i, [tail_10_plus])


    #     first_half_heads = forward_nodes_of_first_half[3][::2]   # mb 0, 1, 2... 9
    #     first_half_tails = forward_nodes_of_first_half[3][1::2]  # mb 0, 1, 2... 9

    #     second_half_heads = forward_nodes[3][::2]                # mb 10, 11, 12... 19
    #     second_half_tails = forward_nodes[3][1::2]     
    #     first_half_bwd_heads = backward_nodes_of_first_half[3][::2]   # mb 0, 1... 9
    #     first_half_bwd_tails = backward_nodes_of_first_half[3][1::2]  # mb 0, 1... 9

    #     second_half_bwd_heads = backward_nodes[3][::2]                # mb 10, 11... 19
    #     second_half_bwd_tails = backward_nodes[3][1::2]     
    #     add_dependencies(second_half_bwd_heads[0], [first_half_tails[4]])
    #     add_dependencies(first_half_bwd_heads[0], [second_half_tails[4]])
    #     add_dependencies(second_half_bwd_heads[1], [first_half_tails[5]])
    #     add_dependencies(first_half_bwd_heads[1], [second_half_tails[5]])
    #     add_dependencies(second_half_bwd_heads[2], [first_half_tails[6]])
    #     add_dependencies(first_half_bwd_heads[2], [second_half_tails[6]])
    #     add_dependencies(second_half_bwd_heads[3], [first_half_tails[7]])
    #     add_dependencies(first_half_bwd_heads[3], [second_half_tails[7]])
    #     add_dependencies(second_half_bwd_heads[4], [first_half_tails[8]])
    #     add_dependencies(first_half_bwd_heads[4], [second_half_tails[8]])
    #     add_dependencies(second_half_bwd_heads[5], [first_half_tails[9]])
    #     add_dependencies(first_half_bwd_heads[5], [second_half_tails[9]])

    #     first_half_heads = forward_nodes_of_first_half[4][::2]   # mb 0, 1, 2... 9
    #     first_half_tails = forward_nodes_of_first_half[4][1::2]  # mb 0, 1, 2... 9

    #     second_half_heads = forward_nodes[4][::2]                # mb 10, 11, 12... 19
    #     second_half_tails = forward_nodes[4][1::2]       
        
    #     first_half_bwd_heads = backward_nodes_of_first_half[4][::2]   # mb 0, 1... 9
    #     first_half_bwd_tails = backward_nodes_of_first_half[4][1::2]  # mb 0, 1... 9

    #     second_half_bwd_heads = backward_nodes[4][::2]                # mb 10, 11... 19
    #     second_half_bwd_tails = backward_nodes[4][1::2]     
    #               # mb 10, 11... 19        # mb 10, 11, 12... 19
    #     add_dependencies(first_half_bwd_heads[0], [second_half_tails[4]])
    #     print(f" ATTENTION!!! Added dependency from {second_half_tails[4].name} to {first_half_bwd_heads[0].name}")
    #     add_dependencies(first_half_tails[4], [first_half_bwd_heads[0]])
    #     add_dependencies(first_half_bwd_heads[1], [second_half_tails[5]])
    #     add_dependencies(second_half_bwd_heads[1], [first_half_tails[5]])
    #     add_dependencies(first_half_bwd_heads[2], [second_half_tails[6]])
    #     add_dependencies(second_half_bwd_heads[2], [first_half_tails[6]])
    #     add_dependencies(first_half_bwd_heads[3], [second_half_tails[7]])
    #     add_dependencies(second_half_bwd_heads[3], [first_half_tails[7]])
    #     add_dependencies(first_half_bwd_heads[4], [second_half_tails[8]])
    #     add_dependencies(second_half_bwd_heads[4], [first_half_tails[8]])
    #     add_dependencies(first_half_bwd_heads[5], [second_half_tails[9]])
    #     add_dependencies(second_half_bwd_heads[5], [first_half_tails[9]])
        return nodes
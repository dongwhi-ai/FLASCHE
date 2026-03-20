`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : InferenceUpdateModule
// Description : Translates the flattened Top-K sorting results into 72-bit 
//               packed words and writes them to the inference memory (CAM). 
//               It manages the memory address mapping based on the target 
//               class, filter ID, and ranking.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module InferenceUpdateModule(
    input wire CLK,
    input wire RSTN,
    input wire start_update,
    input wire [3:0] target_class,       // Target ground-truth class (0-9)
    input wire [3:0] current_filter_id,  // Current filter/kernel ID (0-11)
    
    // Massive flattened input bus containing Top-K results from all 25 parallel patches
    // Dimensions: 25 patches * 32 Top-K ranks * 9-bit pattern indices = 7200 bits
    input wire [25 * 32 * 9 - 1 : 0] sorted_data_flat, 
    
    // --- Inference Memory Write Interface ---
    output reg infer_we,
    output reg [8:0] infer_addr,
    output reg [71:0] infer_data,
    output reg [5:0] infer_sel,
    output reg update_done
);

    localparam integer TOP_K = 32;

    // =========================================================================
    // 1. Data Unpacking (Deserialization)
    // Converts the 1D flat bus into a 2D array structure [Patch][Rank]
    // =========================================================================
    wire [8:0] pat_map [0:24][0:TOP_K-1];
    genvar p, r;
    generate
        for (p=0; p<25; p=p+1) begin : UNPACK_POS
            for (r=0; r<TOP_K; r=r+1) begin : UNPACK_RANK
                assign pat_map[p][r] = sorted_data_flat[(p*TOP_K*9) + (r*9) + 8 : (p*TOP_K*9) + (r*9)];
            end
        end
    endgenerate

    // FSM and Counter Registers
    reg [1:0] state;
    reg [4:0] rank_cnt; // Iterates through the 32 Top-K ranks
    reg [1:0] step_cnt; // Iterates through the 4 write steps per rank
    
    // Array to hold the 25 pattern indices for the currently processing rank
    reg [8:0] curr_rank_pats [0:24];
    integer i;

    // Dynamically select the 25 patterns corresponding to the current rank_cnt
    always @(*) begin
        for(i=0; i<25; i=i+1) curr_rank_pats[i] = pat_map[i][rank_cnt];
    end
    
    // =========================================================================
    // 2. Data Packing Multiplexer
    // Packs the 25 selected 9-bit patterns into four 72-bit words based on step_cnt.
    // Each 72-bit word can hold up to 8 patterns (8 * 9 = 72 bits).
    // =========================================================================
    reg [71:0] packed_data;
    always @(*) begin
        case(step_cnt)
            2'd0: packed_data = {curr_rank_pats[7], curr_rank_pats[6], curr_rank_pats[5], curr_rank_pats[4], 
                                 curr_rank_pats[3], curr_rank_pats[2], curr_rank_pats[1], curr_rank_pats[0]};
            2'd1: packed_data = {curr_rank_pats[15], curr_rank_pats[14], curr_rank_pats[13], curr_rank_pats[12], 
                                 curr_rank_pats[11], curr_rank_pats[10], curr_rank_pats[9], curr_rank_pats[8]};
            2'd2: packed_data = {curr_rank_pats[23], curr_rank_pats[22], curr_rank_pats[21], curr_rank_pats[20], 
                                 curr_rank_pats[19], curr_rank_pats[18], curr_rank_pats[17], curr_rank_pats[16]};
            // The final step only contains the 25th pattern (index 24), padded with zeros.
            2'd3: packed_data = {63'd0, curr_rank_pats[24]};
            default: packed_data = 72'd0;
        endcase
    end

    // =========================================================================
    // 3. Sequential Control Logic (Update FSM)
    // =========================================================================
    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            state       <= 2'd0;
            infer_we    <= 1'b0;
            update_done <= 1'b0;
            rank_cnt    <= 5'd0;
            step_cnt    <= 2'd0;
            infer_addr  <= 9'd0;
            infer_data  <= 72'd0;
            infer_sel   <= 6'd0;
        end else begin
            case (state)
                2'd0: begin // IDLE State
                    update_done <= 1'b0;
                    infer_we    <= 1'b0;
                    if (start_update) begin
                        rank_cnt <= 5'd0;
                        step_cnt <= 2'd0;
                        state    <= 2'd1;
                    end
                end
                
                2'd1: begin // WRITE LOOP State
                    infer_we   <= 1'b1;
                    infer_sel  <= {1'b0, rank_cnt}; // Use rank as the selection index
                    infer_data <= packed_data;
                    
                    // Memory Address Mapping Equation:
                    // 48 words per class (12 filters * 4 steps) + 4 words per filter + current step
                    infer_addr <= (target_class * 9'd48) + (current_filter_id * 9'd4) + {7'd0, step_cnt};
                    
                    // Nested counters for steps (0-3) and ranks (0-31)
                    if (step_cnt == 2'd3) begin
                        step_cnt <= 2'd0;
                        if (rank_cnt == 5'd31) begin
                            state <= 2'd2; // Move to HOLD state after writing all 32 ranks
                        end else begin
                            rank_cnt <= rank_cnt + 5'd1;
                        end
                    end else begin
                        step_cnt <= step_cnt + 2'd1;
                    end
                end
                
                2'd2: begin // DONE & HOLD State
                    infer_we    <= 1'b0;
                    update_done <= 1'b1;
                    // Wait until the control module acknowledges the completion
                    if (!start_update) state <= 2'd0; 
                end
                
                default: state <= 2'd0;
            endcase
        end
    end
endmodule
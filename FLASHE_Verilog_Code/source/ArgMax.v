`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : ArgMax
// Description : A 5-stage pipelined binary tree comparator. It evaluates 10 
//               concurrent input scores to identify the maximum value and its 
//               corresponding index (argmax). The hardcoded pipeline structure 
//               is highly optimized for a 10-class output layer, ensuring a 
//               short critical path and high maximum operating frequency.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module ArgMax #(
    parameter WIDTH = 16,       // Bit-width of each score
    parameter VALS = 10,        // Total number of input values/classes (Fixed to 10 for this logic)
    parameter IDX_WIDTH = 4     // Bit-width required to represent indices 0-9
)(
    input CLK, 
    input en,                   // Pipeline enable and valid data indicator
    input [WIDTH*VALS-1:0] scores, // Packed input bus of 10 scores
    
    output reg [WIDTH-1:0] max_val,  // The highest score found
    output reg [IDX_WIDTH-1:0] max_idx, // The class index of the highest score
    output reg valid                 // High when the final argmax output is ready
    );
    
    // =========================================================================
    // Pipeline Registers
    // =========================================================================
    reg [WIDTH-1:0] v0 [0:VALS-1]; // Stage 0: Latched input array
    reg [4:0] pipe_en = 5'b0;      // Shift register to propagate the enable/valid signal
    
    // Stage 1 Registers (Pairs)
    reg [WIDTH-1:0]     s01_score, s23_score, s45_score, s67_score, s89_score;
    reg [IDX_WIDTH-1:0] s01_idx,   s23_idx,   s45_idx,   s67_idx,   s89_idx;

    // Stage 2 Registers (Quads)
    reg [WIDTH-1:0]     s0123_score, s4567_score;
    reg [IDX_WIDTH-1:0] s0123_idx,   s4567_idx;

    // Stage 3 & 4 Registers (Halves to Final)
    reg [WIDTH-1:0]     s0_7_score, s0_9_score;
    reg [IDX_WIDTH-1:0] s0_7_idx,   s0_9_idx;
    
    // Local constants for index assignments
    localparam [IDX_WIDTH-1:0] IDX_0 = 0, IDX_1 = 1, IDX_2 = 2, IDX_3 = 3,
                               IDX_4 = 4, IDX_5 = 5, IDX_6 = 6, IDX_7 = 7,
                               IDX_8 = 8, IDX_9 = 9;
                               
    integer i;

    // =========================================================================
    // Sequential Logic: Pipelined Tree Comparator
    // =========================================================================
    always @(posedge CLK) begin
        // ---------------------------------------------------------------------
        // Stage 0: Input Latch
        // Unpacks the flattened 160-bit bus into a 10-element array for processing
        // ---------------------------------------------------------------------
        if (en) begin
            for (i = 0; i < VALS; i = i + 1) begin
                v0[i] <= scores[WIDTH*i +: WIDTH];
            end
        end

        // Propagate the pipeline valid signal
        pipe_en <= {pipe_en[3:0], en};

        // ---------------------------------------------------------------------
        // Stage 1: Initial Pairwise Comparisons
        // Reduces 10 candidates down to 5 (0v1, 2v3, 4v5, 6v7, 8v9)
        // ---------------------------------------------------------------------
        if (pipe_en[0]) begin
            {s01_score, s01_idx} <= (v0[0] >= v0[1]) ? {v0[0], IDX_0} : {v0[1], IDX_1};
            {s23_score, s23_idx} <= (v0[2] >= v0[3]) ? {v0[2], IDX_2} : {v0[3], IDX_3};
            {s45_score, s45_idx} <= (v0[4] >= v0[5]) ? {v0[4], IDX_4} : {v0[5], IDX_5};
            {s67_score, s67_idx} <= (v0[6] >= v0[7]) ? {v0[6], IDX_6} : {v0[7], IDX_7};
            {s89_score, s89_idx} <= (v0[8] >= v0[9]) ? {v0[8], IDX_8} : {v0[9], IDX_9};
        end

        // ---------------------------------------------------------------------
        // Stage 2: Quad Comparisons
        // Reduces 4 candidates down to 2 sets (0~3, 4~7). Index 8~9 is held.
        // ---------------------------------------------------------------------
        if (pipe_en[1]) begin
            {s0123_score, s0123_idx} <= (s01_score >= s23_score) ?
                                        {s01_score, s01_idx} : {s23_score, s23_idx};

            {s4567_score, s4567_idx} <= (s45_score >= s67_score) ?
                                        {s45_score, s45_idx} : {s67_score, s67_idx};
        end

        // ---------------------------------------------------------------------
        // Stage 3: Octet Comparison
        // Finds the maximum among indices 0 through 7
        // ---------------------------------------------------------------------
        if (pipe_en[2]) begin
            {s0_7_score, s0_7_idx} <= (s0123_score >= s4567_score) ?
                                      {s0123_score, s0123_idx} : {s4567_score, s4567_idx};
        end

        // ---------------------------------------------------------------------
        // Stage 4: Final Comparison
        // Compares the winner of 0~7 against the winner of 8~9
        // ---------------------------------------------------------------------
        if (pipe_en[3]) begin
            {s0_9_score, s0_9_idx} <= (s0_7_score >= s89_score) ?
                                      {s0_7_score, s0_7_idx} : {s89_score,  s89_idx};
        end

        // ---------------------------------------------------------------------
        // Stage 5: Output Registration
        // Presents the final ArgMax result and asserts the valid flag
        // ---------------------------------------------------------------------
        if (pipe_en[4]) begin
            max_val <= s0_9_score;
            max_idx <= s0_9_idx;
            valid   <= 1'b1;
        end
        else begin
            valid   <= 1'b0;
        end
    end

endmodule
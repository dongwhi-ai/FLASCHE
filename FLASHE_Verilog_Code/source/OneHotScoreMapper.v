`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : OneHotScoreMapper
// Description : Maps a one-hot TCAM match line to a corresponding numerical 
//               score based on the matched rank position. It groups ranks into 
//               score tiers and incorporates a priority encoder as a safety 
//               mechanism against multi-match faults.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module OneHotScoreMapper(
    input CLK, 
    input en, 
    input [`TOPPATTERNS-1:0] match_line, // E.g., 32-bit match line from TCAM
    output reg [15:0] score, 
    output reg valid
    );
    
    reg [15:0] score_temp;
    wire [5:0] score_flags;

    // =========================================================================
    // Combinational Logic: Rank to Score Flag Evaluation
    // [Fix 1] Resolved Vivado VRFC 10-1219 error by utilizing static indexed 
    //         part-selects (+:) instead of dynamic ranges.
    // [Fix 2] Mapped 1-based "Rank < Threshold" conditions to 0-based indexing.
    //         (e.g., Rank 1 corresponds to index 0).
    // =========================================================================
    
    // SCORE1 (Rank < 1): Always false since the minimum rank is 1.
    assign score_flags[0] = 1'b0; 

    // SCORE2 (Rank < 10): Rank 1~9 corresponds to indices 0~8 (9-bit width)
    assign score_flags[1] = |match_line[0 +: 9]; 
    
    // SCORE3 (Rank < 21): Rank 10~20 corresponds to indices 9~19 (11-bit width)
    assign score_flags[2] = |match_line[9 +: 11];
    
    // SCORE4 (Rank < 22): Rank 21 corresponds to index 20 (1-bit width)
    assign score_flags[3] = match_line[20];
    
    // SCORE5 (Rank < 31): Rank 22~30 corresponds to indices 21~29 (9-bit width)
    assign score_flags[4] = |match_line[21 +: 9];
    
    // SCORE6 (Rank < 33): Rank 31~32 corresponds to indices 30~31 (2-bit width)
    assign score_flags[5] = |match_line[30 +: 2];

    // =========================================================================
    // Priority Encoder Safety Net
    // Provides a deterministic fallback in case a corrupted match line contains 
    // multiple high bits, resolving it by prioritizing higher-ranked scores.
    // =========================================================================
    wire [15:0] priority_score = score_flags[1] ? `SCORE2 :
                                 score_flags[2] ? `SCORE3 :
                                 score_flags[3] ? `SCORE4 :
                                 score_flags[4] ? `SCORE5 :
                                 score_flags[5] ? `SCORE6 : 16'd0;

    // =========================================================================
    // Sequential Logic: Output Registration
    // =========================================================================
    always @(posedge CLK) begin
        if (en) score <= score_temp;
        else    score <= 16'd0;
        
        valid <= en;
    end
    
    // =========================================================================
    // Combinational Logic: Score Multiplexer
    // Uses the clean one-hot cases by default, but falls back to the priority 
    // encoder if an abnormal multi-match condition occurs.
    // =========================================================================
    always @(*) begin
        case (score_flags)
            6'b00_0000: score_temp = 16'd0;
            6'b00_0010: score_temp = `SCORE2;
            6'b00_0100: score_temp = `SCORE3;
            6'b00_1000: score_temp = `SCORE4;
            6'b01_0000: score_temp = `SCORE5;
            6'b10_0000: score_temp = `SCORE6;
            default:    score_temp = priority_score; 
        endcase
    end
endmodule
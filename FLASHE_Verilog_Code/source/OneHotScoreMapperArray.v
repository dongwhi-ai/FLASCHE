`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : OneHotScoreMapperArray
// Description : Instantiates an array of 8 parallel OneHotScoreMappers to 
//               evaluate TCAM match lines. It aggregates the individual 
//               scores via a synchronous adder tree to produce a combined 
//               16-bit total score for the current processing cycle.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module OneHotScoreMapperArray(
    input CLK, 
    input [7:0] ens,                          // 8-bit enable vector for each mapper
    input [`TOPPATTERNS*8-1:0] match_lines,   // Packed match lines from 8 TCAM arrays
    output reg [15:0] score,                  // Accumulated total score
    output reg valid                          // High when at least one mapper output is valid
    );
    
    // Internal wires connecting the sub-modules to the adder tree
    wire [16*8-1:0] scores;
    wire [7:0] valids;
    
    // =========================================================================
    // 1. Parallel Score Mapper Instantiation
    // Maps the 8 segments of the packed match_lines bus to individual mappers
    // =========================================================================
    genvar j;
    generate
        for (j=0; j<8; j=j+1) begin: gen
            OneHotScoreMapper OneHotScoreMapper_inst(
                .CLK(CLK), 
                .en(ens[j]), 
                .match_line(match_lines[`TOPPATTERNS*j +: `TOPPATTERNS]), 
                .score(scores[16*j +: 16]), 
                .valid(valids[j])
            );
        end
    endgenerate
    
    // =========================================================================
    // 2. Synchronous Adder Tree
    // Masks invalid scores with zero and sums the valid scores across all 8 channels.
    // =========================================================================
    always @(posedge CLK) begin
        // Bitwise AND each 16-bit score with its replicated valid flag.
        // If valid[j] is 0, the corresponding score contribution is masked to 0.
        score <=
        (scores[16*0 +: 16] & {16{valids[0]}}) +
        (scores[16*1 +: 16] & {16{valids[1]}}) +
        (scores[16*2 +: 16] & {16{valids[2]}}) +
        (scores[16*3 +: 16] & {16{valids[3]}}) +
        (scores[16*4 +: 16] & {16{valids[4]}}) +
        (scores[16*5 +: 16] & {16{valids[5]}}) +
        (scores[16*6 +: 16] & {16{valids[6]}}) +
        (scores[16*7 +: 16] & {16{valids[7]}});
        
        // The global valid signal is high if ANY of the 8 individual valid flags are high (Reduction OR)
        valid <= |(valids);
    end
    
endmodule
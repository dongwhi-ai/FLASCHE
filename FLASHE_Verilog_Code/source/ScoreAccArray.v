`timescale 1ns / 1ps
// `include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : ScoreAccArray
// Description : An array of 10 independent hardware accumulators, typically 
//               representing 10 distinct classification classes (e.g., digits 0-9). 
//               It routes a globally computed score to a specific accumulator 
//               based on a one-hot enable vector and packs the final accumulated 
//               scores into a single 160-bit flat bus.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module ScoreAccArray(
    input CLK, 
    input RSTN, 
    
    // --- Control Signals ---
    input [9:0] ens,             // 10-bit one-hot enable vector (activates specific class accumulator)
    input [9:0] clrs,            // 10-bit clear vector to reset specific accumulators
    input [15:0] acc,            // Common 16-bit score value to be accumulated
    
    // --- Output Ports ---
    output reg [16*10-1:0] scores, // Packed output of all 10 accumulated scores (160 bits)
    output reg valid               // Global valid flag
    );
    
    // Internal wires connecting the 10 individual accumulators
    wire [16*10-1:0] scores_temp;
    wire [9:0] valids_temp;
    
    // =========================================================================
    // 1. Parallel Accumulator Instantiation
    // Generates 10 separate ScoreAcc modules. They all share the same 'acc' 
    // input but are individually controlled via 'ens' and 'clrs' vectors.
    // =========================================================================
    genvar j;
    generate
        for (j=0; j<10; j=j+1) begin: gen
            ScoreAcc ScoreAcc_inst(
                .CLK(CLK), 
                .RSTN(RSTN), 
                .en(ens[j]), 
                .clr(clrs[j]), 
                .acc(acc), 
                .score(scores_temp[16*j +: 16]), 
                .valid(valids_temp[j])
            );
        end
    endgenerate
    
    // =========================================================================
    // 2. Output Combinational Logic
    // Packs the individual score outputs and computes the global valid flag.
    // =========================================================================
    always @(*) begin
        scores = scores_temp;
        
        // Global valid is asserted if ANY of the 10 individual valid flags are high 
        // (Reduction OR operation)
        valid = (|(valids_temp));
    end
    
endmodule
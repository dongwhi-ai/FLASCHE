`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : WeightedNormalizer
// Description : A two-stage pipelined hardware normalizer. It rescales an 
//               input pixel value from an old maximum range to a new maximum 
//               range by multiplying it by NEW_MAX and then dividing by 
//               OLD_MAX. The operation is fully pipelined over two clock cycles.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module WeightedNormalizer #(
    parameter integer NEW_MAX = 16,            // Target maximum value for normalization
    parameter integer OLD_MAX = 48             // Original maximum value of the input data
) (
    input  CLK,
    input  en,                                 // Enable signal for the normalization pipeline
    input  [`PIXELBITS-1:0] idata,             // Input pixel data to be normalized
    output reg valid,                          // High when the normalized output data is valid
    output reg [`PIXELBITS-1:0] odata          // Normalized output pixel data
);

    // =========================================================
    // Internal Registers: Pipeline Stages
    // =========================================================
    reg [`PIXELBITS+2-1:0] mul_result;         // Pipeline register for the multiplication stage
    reg mul_valid;                             // Valid flag for the intermediate multiplication result

    // =========================================================
    // Sequential Logic: 2-Stage Normalization Pipeline
    // Stage 1: Multiply input data by NEW_MAX
    // Stage 2: Divide the multiplied result by OLD_MAX
    // =========================================================
    always @(posedge CLK) begin
        // Shift enable signals through the pipeline to track data validity
        mul_valid <= en;
        valid <= mul_valid;

        // Pipeline Stage 1: Multiplication
        if (en) begin
            mul_result <= idata * NEW_MAX;
        end

        // Pipeline Stage 2: Division
        if (mul_valid) begin
            odata <= mul_result / OLD_MAX;
        end else begin
            odata <= 0;
        end
    end
    
endmodule
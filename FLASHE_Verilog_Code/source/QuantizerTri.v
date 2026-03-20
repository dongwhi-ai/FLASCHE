`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : QuantizerTri
// Description : A 2-bit ternary quantizer. It evaluates an incoming multi-bit 
//               grayscale pixel against two distinct thresholds to categorize 
//               the pixel intensity into one of three states (00, 01, or 10).
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module QuantizerTri(
    input CLK, 
    input en,                          // Enable signal for the quantizer
    input [`PIXELBITS-1:0] idata,      // Input grayscale pixel data
    input [`PIXELBITS-1:0] threshold1, // Lower threshold for ternary quantization
    input [`PIXELBITS-1:0] threshold2, // Upper threshold for ternary quantization
    output reg valid,                  // High when the output data is ready and valid
    output reg [1:0] odata             // 2-bit ternary quantized output (00, 01, or 10)
    );
    
    // Internal wire to hold the combinational comparison result
    reg [1:0] odata_temp;
    
    // =========================================================
    // Sequential Logic: Output Registration
    // Synchronizes the quantized output and valid flag with the clock domain
    // =========================================================
    always @(posedge CLK) begin
        if (en == 1'b1) begin
            valid <= 1'b1;
            odata <= odata_temp;
        end
        else begin
            valid <= 1'b0;
            odata <= 2'b0;
        end
    end
    
    // =========================================================
    // Combinational Logic: Dual-Threshold Comparison
    // - Output 10 (2): Input is greater than or equal to threshold2
    // - Output 01 (1): Input is between threshold1 and threshold2
    // - Output 00 (0): Input is less than threshold1
    // =========================================================
    always @(*) begin
        if (idata >= threshold2) begin
            odata_temp = 2'b10;
        end
        else if (idata >= threshold1) begin
            odata_temp = 2'b01;
        end
        else begin
            odata_temp = 2'b00;
        end
    end
    
endmodule
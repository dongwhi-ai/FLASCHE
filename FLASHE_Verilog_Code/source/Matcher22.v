`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : Matcher22
// Description : A 2x2 pattern matching hardware unit. It compares a 2-bit ternary 
//               quantized patch against a 1-bit binary reference kernel. A key 
//               feature is its wildcard matching capability: if an input pixel 
//               is highly confident (value '10'), it matches any kernel bit.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module Matcher22(
    input CLK, 
    input en,                          // Enable signal for the matching operation
    input [3:0] kernel,                // 4-bit reference kernel (four 1-bit pixels)
    input [2*4-1:0] quantized_patch,   // 8-bit input patch (four 2-bit ternary pixels)
    output reg valid,                  // High when the match result is valid
    output reg match                   // High if the patch matches the kernel
    );
    
    // =========================================================
    // Wire Declarations: Unpacking the Patch and Kernel
    // =========================================================
    wire [1:0] patch_3;
    wire [1:0] patch_2;
    wire [1:0] patch_1;
    wire [1:0] patch_0;
    
    wire kernel_3;
    wire kernel_2;
    wire kernel_1;
    wire kernel_0;
    
    // Unpack the 8-bit quantized patch into four individual 2-bit ternary pixels
    assign patch_3 = quantized_patch[7:6];
    assign patch_2 = quantized_patch[5:4];
    assign patch_1 = quantized_patch[3:2];
    assign patch_0 = quantized_patch[1:0];
    
    // Unpack the 4-bit reference kernel into four individual 1-bit pixels
    assign kernel_3 = kernel[3];
    assign kernel_2 = kernel[2];
    assign kernel_1 = kernel[1];
    assign kernel_0 = kernel[0];
    
    // Internal wire for the combinational match result
    reg match_temp;
    
    // =========================================================
    // Sequential Logic: Output Registration
    // Synchronizes the matching result and valid flag with the clock domain
    // =========================================================
    always @(posedge CLK) begin
        if (en == 1'b1) begin
            valid <= 1'b1;
            match <= match_temp;
        end
        else begin
            valid <= 1'b0;
            match <= 1'b0;
        end
    end
    
    // =========================================================
    // Combinational Logic: Wildcard Pattern Matching
    // Evaluates pixel-by-pixel. A match occurs if the patch pixel 
    // exactly equals the kernel bit (implicitly padded to 2 bits) 
    // OR if the patch pixel is '10' (acts as a wildcard/don't care).
    // All four pixels must meet this condition simultaneously.
    // =========================================================
    always @(*) begin
        match_temp = (patch_3 == kernel_3 || patch_3 == 2'b10) && 
                     (patch_2 == kernel_2 || patch_2 == 2'b10) && 
                     (patch_1 == kernel_1 || patch_1 == 2'b10) && 
                     (patch_0 == kernel_0 || patch_0 == 2'b10);
    end
    
endmodule
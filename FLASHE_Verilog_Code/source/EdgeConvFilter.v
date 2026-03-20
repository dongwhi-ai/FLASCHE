`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : EdgeConvFilter
// Description : A conditional convolution filter module for a 2x2 patch. 
//               It performs a multiply-accumulate (MAC) operation between 
//               a multi-bit image patch and a 1-bit reference kernel. 
//               The convolution is conditionally executed only if a valid 
//               pattern match signal is asserted.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module EdgeConvFilter(
    input CLK, 
    input en,                                  // Enable signal for the filter operation
    input [`PIXELBITS*4-1:0] image_patch,      // 4-pixel image patch (each pixel is `PIXELBITS wide)
    input match,                               // Match flag from the preceding pattern matcher
    input [3:0] kernel,                        // 4-bit reference kernel (four 1-bit pixels)
    output reg valid,                          // High when the convolution output data is valid
    output reg [`PIXELBITS-1:0] odata          // Accumulated output data (convolution result)
    );
    
    // =========================================================
    // Wire Declarations: Unpacking the Patch and Kernel
    // =========================================================
    wire [`PIXELBITS-1:0] data_3;
    wire [`PIXELBITS-1:0] data_2;
    wire [`PIXELBITS-1:0] data_1;
    wire [`PIXELBITS-1:0] data_0;
    
    wire kernel_3;
    wire kernel_2;
    wire kernel_1;
    wire kernel_0;
    
    // Unpack the incoming multi-bit image patch into four individual pixels
    assign data_3 = image_patch[`PIXELBITS*4-1:`PIXELBITS*3];
    assign data_2 = image_patch[`PIXELBITS*3-1:`PIXELBITS*2];
    assign data_1 = image_patch[`PIXELBITS*2-1:`PIXELBITS];
    assign data_0 = image_patch[`PIXELBITS-1:0];
    
    // Unpack the 4-bit reference kernel into four individual 1-bit pixels
    assign kernel_3 = kernel[3];
    assign kernel_2 = kernel[2];
    assign kernel_1 = kernel[1];
    assign kernel_0 = kernel[0];
    
    // Internal register for the combinational MAC result
    reg [`PIXELBITS-1:0] odata_temp;
    
    // =========================================================
    // Sequential Logic: Output Registration
    // Synchronizes the output data and valid flag with the clock domain
    // =========================================================
    always @(posedge CLK) begin
        if (en == 1'b1) begin
            valid <= 1'b1;
            odata <= odata_temp;
        end
        else begin
            valid <= 1'b0;
            odata <= 0;
        end
    end
    
    // =========================================================
    // Combinational Logic: Conditional Convolution (MAC)
    // Computes the dot product of the kernel and the image patch.
    // The operation is gated by the 'match' signal.
    // =========================================================
    always @(*) begin
        if (match == 1'b1) begin
            odata_temp = (kernel_3 * data_3) + 
                         (kernel_2 * data_2) + 
                         (kernel_1 * data_1) + 
                         (kernel_0 * data_0);
        end
        else begin
            odata_temp = 0;
        end
    end
    
endmodule
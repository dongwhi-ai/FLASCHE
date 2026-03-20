`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : QImageBuffer
// Description : A wrapper module integrating a binary quantizer and an image 
//               buffer. It streams multi-bit pixels, converts them to 1-bit 
//               binary values based on a defined threshold, and stacks them 
//               to construct a complete, flattened binary image array.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module QImageBuffer(
    input CLK, 
    input stack_en,                 // Enable signal for incoming pixels
    input [`PIXELBITS-1:0] pixel,   // Multi-bit input pixel data
    output reg image_valid,         // High when the full image buffer is completely filled
    output reg [`IMAGEARR2-1:0] image // Flattened 1-bit binary image array output
    );
    
    // Threshold parameter for binary quantization
    parameter [`PIXELBITS-1:0] bi_threshold = `BITHRSCR;
    
    // =========================================================================
    // Internal Wires and Pipeline Registers
    // =========================================================================
    reg stack_en_d1; // 1-cycle delayed enable to synchronize with quantizer latency
    
    wire image_valid_inst;
    wire [`IMAGEARR2-1:0] image_inst;
    
    wire quant_out;
    wire quant_valid; // Wire to catch the quantizer's valid output
    
    // =========================================================================
    // 1. Binary Quantizer Instantiation
    // Converts the incoming multi-bit pixel into a 1-bit value (0 or 1).
    // Introduces a 1-clock cycle latency.
    // =========================================================================
    QuantizerBi QuantizerBi_inst(
        .CLK(CLK), 
        .en(stack_en), 
        .idata(pixel), 
        .threshold(bi_threshold), 
        .valid(quant_valid), 
        .odata(quant_out)
    );
    
    // =========================================================================
    // 2. Image Buffer Instantiation
    // Accumulates the 1-bit quantized pixels into a full image array.
    // =========================================================================
    ImageBuffer ImageBuffer_inst(
        .CLK(CLK), 
        // Feed the delayed enable to align with the quantized pixel arrival
        .stack_en(stack_en_d1), 
        .pixel(quant_out), 
        .image_valid(image_valid_inst), 
        .image(image_inst)
    );
    
    // =========================================================================
    // Sequential Logic: Pipeline Synchronization
    // Delays the stack_en signal by 1 clock cycle to match the quantizer latency.
    // =========================================================================
    always @(posedge CLK) begin
        if (stack_en == 1'b1) begin
            stack_en_d1 <= 1'b1;
        end
        else begin
            stack_en_d1 <= 1'b0;
        end
    end
    
    // =========================================================================
    // Combinational Logic: Output Routing
    // =========================================================================
    always @(*) begin
        image_valid = image_valid_inst;
        image = image_inst;
    end
    
endmodule
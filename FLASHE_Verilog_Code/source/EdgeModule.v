`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : EdgeModule
// Description : Executes a parallel sliding-window edge and feature extraction
//               pipeline. It utilizes ping-pong line buffers to form 2x2 patches, 
//               performs ternary quantization, checks for pattern matches across 
//               15 distinct kernels, applies convolution, and normalizes the outputs.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module EdgeModule(
    input CLK,
    input RSTN, 
    input en,                       // Enable signal for the processing pipeline
    input use_match,                // Flag to enable/disable early pattern matching
    input [`PIXELBITS-1:0] idata,   // Input pixel stream
    output reg [14:0] valids,       // 15-bit valid flags for each parallel kernel output
    output reg [`PIXELBITS*15-1:0] odatas // Packed output array of 15 feature maps
    );
    
    // =========================================================
    // Pipeline Parameters & Constants
    // =========================================================
    parameter [`PIXELBITS-1:0] tri_threshold1 = `TRITHR1; // Ternary quantization lower threshold
    parameter [`PIXELBITS-1:0] tri_threshold2 = `TRITHR2; // Ternary quantization upper threshold
    parameter [`PIXELBITS-1:0] prelayer_max = `PRELAYERMAX; // Normalization constant
    parameter [`PIXELBITS-1:0] new_max = `NEWMAX;           // Target normalization max value
    
    // Packed array of 15 pre-defined 4-bit 2x2 edge kernels (15 * 4 = 60 bits)
    parameter [59:0] kernels = 60'b1111_1110_1101_1100_1011_1010_1001_1000_0111_0110_0101_0100_0011_0010_0001;
    
    // Pre-computed scaling configurations for the normalizer
    parameter [(`PIXELBITS+2)*15-1:0] old_maxs_conf =
        { 10'd64,10'd48,10'd48,10'd32,10'd48,10'd32,10'd32,10'd16,
          10'd48,10'd32,10'd32,10'd16,10'd32,10'd16,10'd16};
    
    // =========================================================
    // Internal Registers & Line Buffers
    // =========================================================
    wire compile;
    reg compile_d1;
    reg [`PIXELBITS-1:0] idata_d1;
    reg en_d1;
    assign compile = en && !en_d1; // Rising edge detector for enable signal
    
    reg [3:0] kernels_reg [14:0];
    
    // Ping-pong line buffers to store adjacent rows for 2x2 patch extraction
    reg [`PIXELBITS-1:0] image_storage1 [`IMAGELEN1-1:0];
    reg [`PIXELBITS-1:0] image_storage2 [`IMAGELEN1-1:0];
    reg [1:0] quant_storage1 [`IMAGELEN1-1:0];
    reg [1:0] quant_storage2 [`IMAGELEN1-1:0];
    reg [3:0] weights [14:0];
    
    // Pipeline column pointers and row toggle flags
    reg [7:0] col;
    reg [7:0] match_col;
    reg [7:0] conv_col;
    reg lowhigh;           // Toggles between image_storage1 and image_storage2
    reg match_lowhigh;
    reg conv_lowhigh;
    reg patch_ready;       // High when a full 2x2 patch is available
    
    // =========================================================
    // Stage 1: Ternary Quantizer
    // Converts input grayscale pixels into 2-bit ternary values
    // =========================================================
    wire quant_valid;
    wire [1:0] quant_out;
    QuantizerTri QuantizerTri_inst(
        .CLK(CLK), 
        .en(en_d1), 
        .idata(idata), 
        .threshold1(tri_threshold1), 
        .threshold2(tri_threshold2), 
        .valid(quant_valid), 
        .odata(quant_out)
    );
    
    // =========================================================
    // Stage 2: Parallel Pattern Matching
    // =========================================================
    reg match_en;
    wire [2*4-1:0] quantized_patch;
    
    // Multiplexes the 2x2 patch dynamically based on the current active line buffer
    assign quantized_patch = (match_lowhigh==1'b1) ? 
           {quant_storage1[match_col], quant_storage1[match_col+1], quant_storage2[match_col], quant_storage2[match_col+1]} : 
           {quant_storage2[match_col], quant_storage2[match_col+1], quant_storage1[match_col], quant_storage1[match_col+1]};
           
    wire [14:0] match_valids;
    wire match_valid;
    assign match_valid = &match_valids; // Global valid when all 15 matchers complete
    wire [14:0] match_outs;
    
    genvar j;
    generate
        for (j=0; j<15; j=j+1) begin: gen0
            Matcher22 Matcher22_inst(
                .CLK(CLK), 
                .en(match_en), 
                .kernel(kernels_reg[j]), 
                .quantized_patch(quantized_patch), 
                .valid(match_valids[j]), 
                .match(match_outs[j])
            );
        end
    endgenerate
    
    // =========================================================
    // Stage 3: Edge Convolution Filters
    // Executes 2D convolution only on patches that triggered a match
    // =========================================================
    wire [8*4-1:0] image_patch;
    assign image_patch = (conv_lowhigh==1'b1) ? 
           {image_storage1[conv_col], image_storage1[conv_col+1], image_storage2[conv_col], image_storage2[conv_col+1]} : 
           {image_storage2[conv_col], image_storage2[conv_col+1], image_storage1[conv_col], image_storage1[conv_col+1]};
           
    wire [14:0] conv_valids;
    wire conv_valid;
    assign conv_valid = &(conv_valids);
    wire [`PIXELBITS-1:0] conv_outs [14:0];  
    
    generate
        for (j=0; j<15; j=j+1) begin: gen1
            EdgeConvFilter EdgeConvFilter_inst(
                .CLK(CLK), 
                .en(match_valid), 
                .image_patch(image_patch), 
                .match(match_outs[j] || !use_match), // Bypass match condition if use_match is disabled
                .kernel(kernels_reg[j]), 
                .valid(conv_valids[j]), 
                .odata(conv_outs[j])
            );
        end
    endgenerate
    
    // =========================================================
    // Stage 4: Weighted Normalizers
    // Scales the convolution outputs uniformly
    // =========================================================
    wire [14:0] norm_valids;
    wire [`PIXELBITS-1:0] norm_outs [14:0];
    reg [`PIXELBITS+2-1:0] old_maxs [14:0];

    generate
        for (j=0; j<15; j=j+1) begin: gen2
            WeightedNormalizer #(
                .NEW_MAX(new_max),  
                .OLD_MAX(old_maxs_conf[j*10 +: 10]) 
            ) WeightedNormalizer_inst (
                .CLK(CLK),
                .en(conv_valid),
                .idata(conv_outs[j]),
                .valid(norm_valids[j]),
                .odata(norm_outs[j])
            );
        end
    endgenerate

    // =========================================================
    // Main Sequential Logic: Pipeline Control & Buffering
    // =========================================================
    integer i;
    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            compile_d1 <= 0;
            idata_d1 <= 0;
            en_d1 <= 0;
            for (i=0; i<15; i=i+1) begin
                kernels_reg[i] <= 0;
                weights[i] <= 0;
                old_maxs[i] <= 0;
            end
            for (i=0; i<`IMAGELEN1; i=i+1) begin
                image_storage1[i] <= 0;
                image_storage2[i] <= 0;
                quant_storage1[i] <= 0;
                quant_storage2[i] <= 0;
            end
            col <= 0;
            match_col <= 0;
            conv_col <= 0;
            lowhigh <= 0;
            match_lowhigh <= 0;
            conv_lowhigh <= 0;
        end
        else begin
            // Shift pipeline registers
            compile_d1 <= compile;
            idata_d1 <= idata;
            en_d1 <= en;
            
            // Initialization: Extract 4-bit kernels and calculate structural weights
            if (compile==1'b1) begin
                for (i=0; i<15; i=i+1) begin
                    kernels_reg[i] <= kernels[i*4 +: 4];
                    weights[i] <= kernels[i*4+3] + kernels[i*4+2] + kernels[i*4+1] + kernels[i*4];
                end
            end
            
            if (compile_d1==1'b1) begin
                for (i=0; i<15; i=i+1) begin
                    old_maxs[i] <= prelayer_max * weights[i];
                end
            end
            
            // Pipeline propagation of the row toggle flags
            match_lowhigh <= lowhigh;
            conv_lowhigh <= match_lowhigh;
            
            // Manage Ping-Pong Line Buffers
            if (quant_valid==1'b1) begin
                if (col==`IMAGELEN1-1) begin
                    col <= 0;
                    lowhigh <= ~lowhigh; // Toggle row buffer upon completing a row
                    if (patch_ready==1'b0) begin
                        patch_ready <= 1'b1; // First row complete, 2x2 patch can now form
                    end
                end
                else begin
                    col <= col+1;
                end
                
                // Route incoming pixel and quantized data to the active buffer
                if (lowhigh==1'b0) begin
                    image_storage1[col] <= idata_d1;
                    quant_storage1[col] <= quant_out;
                end
                else begin
                    image_storage2[col] <= idata_d1;
                    quant_storage2[col] <= quant_out;
                end
            end
            else begin
                col <= 0;
                lowhigh <= 0;
                patch_ready <= 1'b0;
            end
            
            // Control matching pipeline bounds
            if (patch_ready==1'b1) begin
                if (col==0) begin
                    match_en <= 1'b0; // Pause at row boundaries to prevent wrap-around
                end
                else begin
                    match_en <= 1'b1;
                end 
            end
            
            // Update sliding window column pointer for matching phase
            if (match_en==1'b1) begin
                if (match_col==`IMAGELEN1-2) begin
                    match_col <= 0;
                end
                else begin
                    match_col <= match_col + 1;
                end
            end
            
            // Synchronize convolution column pointer with matching pointer
            conv_col <= match_col;
        end
    end
    
    // =========================================================
    // Output Combinational Logic
    // Packs the 15 normalized parallel outputs into a single bus
    // =========================================================
    always @(*) begin
        valids = norm_valids; // Assign validation signal based on normalizer completion
        for (i=0; i<15; i=i+1) begin
            odatas[i*`PIXELBITS +: `PIXELBITS] = norm_outs[i];
        end
    end
    
endmodule
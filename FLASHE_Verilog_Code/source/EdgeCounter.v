`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Module: EdgeCounter
// Description: Extracts 2x2 binary patches from an incoming 8x8 pixel stream,
//              builds a histogram of the 16 possible edge patterns, and 
//              identifies the most frequent (top-k) patterns.
// ============================================================================
module EdgeCounter(
    input CLK, 
    input en,                     // Enable signal for pixel processing
    input [`PIXELBITS-1:0] pixel, // Input pixel data
    input top_filter_en,          // Trigger signal to evaluate and output top-k mask
    output reg [15:0] top_mask,   // 16-bit mask representing the selected top-k patterns
    output reg top_mask_valid     // High when the top_mask output is valid
    );
    
    // Threshold for binary quantization (defined in defs.vh)
    parameter [`PIXELBITS-1:0] bi_threshold = `BITHRTOP;
    
    // Histogram counters for the 16 possible 2x2 binary patterns (0000 to 1111)
    reg [15:0] filter_counts [15:0];
    
    // 8x8 buffer to store the binarized image frame
    reg [7:0] bin_img [7:0];
    
    // 4-bit wire representing the current 2x2 sliding window patch
    wire [3:0] quant_patch;
    
    // Image coordinate trackers for the 8x8 block
    reg [7:0] pixel_col_counter = 0;
    reg [7:0] pixel_row_counter = 0;
    
    // =========================================================
    // 1. Binarization Stage (Quantizer)
    // =========================================================
    wire quant_valid;
    wire quant_out;
    QuantizerBi QuantizerBi_inst(
        .CLK(CLK), 
        .en(en), 
        .idata(pixel), 
        .threshold(bi_threshold), 
        .valid(quant_valid), 
        .odata(quant_out) // 1-bit binarized pixel output
    );
    
    // =========================================================
    // 2. 2x2 Patch Extraction (Sliding Window)
    // =========================================================
    
    // Concatenates adjacent pixels from the buffer and the current input
    // to form a 4-bit edge pattern (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    assign quant_patch = {bin_img[pixel_row_counter-1][pixel_col_counter-1], 
                          bin_img[pixel_row_counter-1][pixel_col_counter], 
                          bin_img[pixel_row_counter][pixel_col_counter-1], 
                          quant_out};
    
    // =========================================================
    // 3. Top-k Mask Extraction
    // =========================================================
    wire [15:0] top_mask_temp; 
    wire top_mask_valid_temp;
    
    // Sub-module that sorts the histogram and outputs a 16-bit mask of the highest counts
    TopkMask TopkMask_inst(
        .CLK(CLK), 
        .en(top_filter_en), 
        .counts({filter_counts[15], filter_counts[14], filter_counts[13], filter_counts[12], 
                 filter_counts[11], filter_counts[10], filter_counts[9], filter_counts[8], 
                 filter_counts[7], filter_counts[6], filter_counts[5], filter_counts[4], 
                 filter_counts[3], filter_counts[2], filter_counts[1], filter_counts[0]}), 
        .top_valid(top_mask_valid_temp), 
        .top_mask(top_mask_temp)
    );
    
    always @(*) begin
        top_mask = top_mask_temp;
        top_mask_valid = top_mask_valid_temp;
    end
    
    // =========================================================
    // 4. Sequential Logic: Coordinate Tracking & Histogram Update
    // =========================================================
    integer i;
    always @(posedge CLK) begin
        if (top_mask_valid==1'b0) begin
            if (quant_valid==1'b1) begin
                // Update column and row counters for an 8x8 image block
                if (pixel_row_counter==7) begin
                    if (pixel_col_counter==7) begin
                        pixel_col_counter <= 0;
                        pixel_row_counter <= 0;
                    end
                    else begin
                        pixel_col_counter <= pixel_col_counter + 1;
                    end
                end
                else begin
                    if (pixel_col_counter==7) begin
                        pixel_col_counter <= 0;
                        pixel_row_counter <= pixel_row_counter + 1;
                    end
                    else begin
                        pixel_col_counter <= pixel_col_counter + 1;
                    end
                end
                
                // Store the incoming binarized pixel into the frame buffer
                bin_img[pixel_row_counter][pixel_col_counter] <= quant_out;
            end
            
            // Wait until at least the 2nd row and 2nd column are reached
            // to form a valid 2x2 patch, then update the corresponding histogram bin
            if (pixel_row_counter >= 1) begin
                if (pixel_col_counter >= 1) begin
                    if (quant_patch!=0) begin // Ignore completely blank (0000) patches
                        filter_counts[quant_patch] <= filter_counts[quant_patch] + 1;
                    end
                end
            end
        end
        else begin
            // Reset all histogram counters when the top mask is valid/extracted
            for (i=0; i<16; i=i+1) begin
                filter_counts[i] <= 0;
            end
        end
    end
    
endmodule
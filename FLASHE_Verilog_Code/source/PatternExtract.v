`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : PatternExtract
// Description : A feature extraction module that applies a 3x3 sliding window 
//               over a 7x7 binary image. It extracts 25 overlapping 3x3 patches 
//               and packs them into a single parallel bus for downstream processing.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module PatternExtract(
    input CLK, 
    input en,                       // Enable signal for the extraction process
    input [48:0] image_flat,        // Flattened 7x7 binary image input (49 bits)
    output reg [9*25-1:0] patterns, // Packed output of 25 patches (9 bits each)
    output reg valid                // High when the output patterns are valid
);
    
    // Internal 2D array to hold the 25 combinationally extracted 3x3 patches
    reg [8:0] patterns_temp [24:0];
    
    // =========================================================================
    // Sequential Logic: Output Registration & Packing
    // Packs the 25 individual 9-bit patches into a single 225-bit flat bus
    // =========================================================================
    always @(posedge CLK) begin
        if (en == 1'b1) begin
            patterns <= { patterns_temp[24],
                          patterns_temp[23],
                          patterns_temp[22],
                          patterns_temp[21],
                          patterns_temp[20],
                          patterns_temp[19],
                          patterns_temp[18],
                          patterns_temp[17],
                          patterns_temp[16],
                          patterns_temp[15],
                          patterns_temp[14],
                          patterns_temp[13],
                          patterns_temp[12],
                          patterns_temp[11],
                          patterns_temp[10],
                          patterns_temp[9],
                          patterns_temp[8],
                          patterns_temp[7],
                          patterns_temp[6],
                          patterns_temp[5],
                          patterns_temp[4],
                          patterns_temp[3],
                          patterns_temp[2],
                          patterns_temp[1],
                          patterns_temp[0] };
            valid <= 1'b1;
        end
        else begin
            patterns <= { (9*25){1'b0} }; // Zero-fill the bus if not enabled
            valid <= 1'b0;
        end
    end
    
    // =========================================================================
    // Combinational Logic: Sliding Window Extraction
    // Iterates through a 5x5 grid representing the valid top-left anchor points 
    // for a 3x3 window within a 7x7 image. 
    // The 1D index mapping formula is: Width * (Row_Anchor + y_offset) + (Col_Anchor + x_offset)
    // =========================================================================
    integer r, c;
    always @(*) begin
        // r: Row anchor (0 to 4)
        for (r = 0; r < 5; r = r + 1) begin
            // c: Column anchor (0 to 4)
            for (c = 0; c < 5; c = c + 1) begin
                
                // Map the 3x3 pixels into a 9-bit vector
                patterns_temp[r*5 + c] = {
                    image_flat[7*(r+0) + (c+0)], // Top-Left
                    image_flat[7*(r+0) + (c+1)], // Top-Center
                    image_flat[7*(r+0) + (c+2)], // Top-Right
                    
                    image_flat[7*(r+1) + (c+0)], // Mid-Left
                    image_flat[7*(r+1) + (c+1)], // Center
                    image_flat[7*(r+1) + (c+2)], // Mid-Right
                    
                    image_flat[7*(r+2) + (c+0)], // Bot-Left
                    image_flat[7*(r+2) + (c+1)], // Bot-Center
                    image_flat[7*(r+2) + (c+2)]  // Bot-Right
                };
            end
        end
    end

endmodule
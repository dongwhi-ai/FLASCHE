`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : ImageBuffer
// Description : A serial-in, parallel-out (SIPO) buffer for binary image data. 
//               It sequentially stacks incoming 1-bit pixels into a flattened 
//               register array. Once the array is fully populated to the 
//               target dimension, it asserts a valid signal to trigger 
//               downstream inference processing.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module ImageBuffer(
    input CLK, 
    input stack_en,                  // Enable signal to push a new pixel into the buffer
    input pixel,                     // 1-bit input pixel (from the binarization quantizer)
    
    output reg image_valid,          // High for one cycle when the full image buffer is ready
    output reg [`IMAGEARR2-1:0] image // Flattened parallel output of the accumulated image
    );
    
    // Internal counter to track the number of accumulated pixels
    reg [7:0] counter = 0;
    
    // =========================================================================
    // Sequential Logic: Accumulation and Output Trigger
    // =========================================================================
    always @(posedge CLK) begin
        if (stack_en) begin
            // Check if this is the last pixel needed to complete the image array
            if (counter == `IMAGEARR2-1) begin
                image_valid    <= 1'b1;     // Assert valid flag indicating image is complete
                counter        <= 0;        // Reset the counter for the next image frame
                image[counter] <= pixel;    // Latch the final pixel into the highest index
            end
            else begin
                image_valid    <= 1'b0;     // Image is not yet complete
                counter        <= counter + 1; // Increment the pixel index
                image[counter] <= pixel;    // Stack the incoming pixel at the current index
            end
        end
        else begin
            // If stacking is disabled, ensure valid signal remains low
            image_valid <= 1'b0;
        end
    end
    
endmodule
`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Module: QuantizerBi
// Description: A simple 1-bit quantizer (binarizer) that converts an 8-bit 
//              grayscale input pixel into a 1-bit output based on a given 
//              threshold value.
// Tool       : Xilinx Vivado 2024.2
// ============================================================================
module QuantizerBi(
    input CLK, 
    input en,              // Enable signal for quantization
    input [7:0] idata,     // 8-bit input grayscale pixel data
    input [7:0] threshold, // 8-bit threshold value for binarization
    output reg valid,      // High when the output data (odata) is valid
    output reg odata       // 1-bit binarized output pixel
    );
    
    // Internal wire for combinational comparison result
    reg odata_temp;
    
    // =========================================================
    // Sequential Logic: Output Registration
    // Synchronizes the binarized output and valid flag with the clock
    // =========================================================
    always @(posedge CLK) begin
        if (en==1'b1) begin
            valid <= 1'b1;
            odata <= odata_temp; 
        end
        else begin
            valid <= 1'b0;
            odata <= 1'b0;
        end
    end
    
    // =========================================================
    // Combinational Logic: Threshold Comparison
    // Sets the output to 1 if the input pixel is greater than or 
    // equal to the threshold, otherwise 0.
    // =========================================================
    always @(*) begin
        if (idata >= threshold) begin
            odata_temp = 1'b1;
        end
        else begin
            odata_temp = 1'b0;
        end
    end
    
endmodule
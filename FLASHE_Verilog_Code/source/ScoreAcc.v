`timescale 1ns / 1ps
// `include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : ScoreAcc
// Description : A 16-bit synchronous hardware accumulator. It adds the incoming 
//               score to its internal register when enabled, holds the value 
//               otherwise, and can be synchronously cleared for a new inference.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module ScoreAcc(
    input CLK, 
    input RSTN,              // Active-low asynchronous reset
    input en,                // Enable signal to trigger accumulation
    input clr,               // Synchronous clear to reset the score to zero
    input [15:0] acc,        // 16-bit incoming score to be added
    
    output reg [15:0] score, // 16-bit accumulated total score
    output reg valid         // High for one cycle when an accumulation occurs
    );
    
    // =========================================================================
    // Sequential Logic: Accumulator Datapath
    // Combines an adder and a feedback register to maintain the running total.
    // =========================================================================
    always @(posedge CLK) begin
        // 1. Asynchronous Reset (System Initialization)
        if (!RSTN) begin
            score <= 16'd0;
            valid <= 1'b0;
        end
        else begin
            // 2. Accumulate Operation
            if (en == 1'b1) begin
                score <= score + acc; // Add incoming value to the running total
                valid <= 1'b1;        // Pulse valid to indicate an update occurred
            end
            // 3. Synchronous Clear Operation
            else if (clr == 1'b1) begin
                score <= 16'd0;       // Reset the accumulator for the next batch/class
                valid <= 1'b0;
            end
            // 4. Hold State
            else begin
                // score retains its previous value implicitly
                valid <= 1'b0;        // Valid drops low as no new addition happened
            end
        end
    end
    
endmodule
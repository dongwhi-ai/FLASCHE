`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : SortCell
// Description : A single compare-and-swap (CAS) processing element within the 
//               systolic array sorter. It compares incoming data against its 
//               stored value, retaining the larger one and pushing the smaller 
//               one to the adjacent downstream cell in the pipeline.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module SortCell(
    input CLK,
    input RSTN,
    input soft_reset,               // Synchronous reset to clear data between different filters
    input en,                       // Pipeline enable signal
    
    // --- Input from the upstream (previous) cell ---
    input [7:0] in_count,           // 8-bit occurrence count (frequency, Max 255)
    input [8:0] in_pattern_idx,     // 9-bit pattern identifier
    
    // --- Output to the downstream (next) cell ---
    output reg [7:0] out_count,     // Count value pushed down the array
    output reg [8:0] out_pattern_idx, // Pattern ID pushed down the array
    
    // --- Internal State (Locally Stored Result) ---
    output reg [7:0] my_count,      // The count currently retained by this cell
    output reg [8:0] my_pattern_idx // The pattern ID currently retained by this cell
    );

    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            my_count        <= 8'd0;
            my_pattern_idx  <= 9'd0;
            out_count       <= 8'd0;
            out_pattern_idx <= 9'd0;
        end else if (soft_reset) begin
            my_count        <= 8'd0;
            my_pattern_idx  <= 9'd0;
            out_count       <= 8'd0;
            out_pattern_idx <= 9'd0;
        end else if (en) begin
            // =================================================================
            // Core Compare-and-Swap (CAS) Logic with Tie-Breaker
            // The cell updates its stored value if:
            // 1. The incoming count is strictly greater than the stored count.
            // 2. The counts are equal (and > 0), but the incoming pattern index 
            //    is larger. This tie-breaking condition ensures deterministic 
            //    and stable sorting behavior.
            // =================================================================
            if ((in_count > my_count) || (in_count > 0 && in_count == my_count && in_pattern_idx > my_pattern_idx)) begin
                // Retain the incoming larger (or winning tie) data
                my_count        <= in_count;
                my_pattern_idx  <= in_pattern_idx;
                
                // Evict the previously stored data, pushing it to the next cell
                out_count       <= my_count;
                out_pattern_idx <= my_pattern_idx;
            end else begin
                // The incoming data is smaller or null (0); pass it straight through
                // without modifying this cell's internal state.
                out_count       <= in_count;
                out_pattern_idx <= in_pattern_idx;
            end
        end
    end
endmodule
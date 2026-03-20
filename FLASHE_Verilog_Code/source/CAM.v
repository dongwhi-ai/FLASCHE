`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : CAM
// Description : A single 9-bit Ternary Content-Addressable Memory (CAM) cell.
//               It stores a programmed pattern and compares it against a 
//               target search pattern in a single clock cycle. It features a 
//               ternary "don't-care" mask (xmask) that can force a bitwise 
//               match regardless of the stored or target bit values.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module CAM(
    input CLK, 
    
    // --- Programming Interface ---
    input set,                   // Write enable: loads the set_pattern into the cell
    input [8:0] set_pattern,     // The 9-bit pattern to be programmed and stored
    
    // --- Search Interface ---
    input en,                    // Search enable: triggers the match evaluation
    input [8:0] target_pattern,  // The 9-bit key to search for
    input [8:0] xmask,           // Ternary mask (1 = Don't care, 0 = Exact match required)
    
    // --- Output Results ---
    output reg match,            // High if the target matches the stored pattern (considering the mask)
    output reg valid             // High when the search operation is complete and 'match' is valid
    );
    
    // =========================================================================
    // Internal Registers
    // =========================================================================
    reg [8:0] saved_pattern = 0; // Holds the programmed Top-K pattern
    reg [8:0] pass = 0;          // Intermediate wire-like reg for bitwise match results
    reg match_temp = 0;          // Combinational final match result
    
    // =========================================================================
    // Sequential Logic: Pattern Storage & Output Registration
    // =========================================================================
    always @(posedge CLK) begin
        // Program the cell with a new pattern when 'set' is high
        if (set == 1'b1) begin
            saved_pattern <= set_pattern;
        end
        
        // Register the match result when the search is enabled
        if (en == 1'b1) begin
            match <= match_temp;
            valid <= 1'b1;
        end
        else begin
            valid <= 1'b0;
        end
    end
    
    // =========================================================================
    // Combinational Logic: Ternary Match Evaluation
    // =========================================================================
    integer i;
    always @(*) begin
        // 1. Bitwise Comparison with Don't-Care Masking
        for (i=0; i<9; i=i+1) begin
            // A bit passes if the mask explicitly ignores it (xmask == 1) 
            // OR if the target bit perfectly matches the stored bit.
            if (xmask[i] == 1'b1 || target_pattern[i] == saved_pattern[i]) begin
                pass[i] = 1'b1;
            end
            else begin
                pass[i] = 1'b0;
            end
        end
        
        // 2. Global Match Resolution
        // The overall cell matches ONLY if all 9 individual bits report a 'pass'.
        if (pass == 9'b111111111) begin
            match_temp = 1'b1;
        end
        else begin
            match_temp = 1'b0;
        end
    end
    
endmodule
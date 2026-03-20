`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : CAMArray
// Description : A highly parallel Ternary Content-Addressable Memory (CAM) 
//               array. It instantiates multiple individual CAM cells to perform 
//               a single-cycle, simultaneous search of a target pattern (with 
//               don't-care masking) against all stored Top-K patterns.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module CAMArray(
    input CLK, 
    
    // --- Programming (Write) Interface ---
    input [9*32-1:0] set_patterns,   // Packed bus of 32 patterns (9 bits each) to initialize the CAM
    input set,                       // Write enable signal to program the patterns into the cells
    
    // --- Search (Read/Match) Interface ---
    input en,                        // Enable signal to start the search operation
    input [8:0] target_pattern,      // The 9-bit search key (feature extracted from the image)
    input [8:0] xmask,               // Ternary mask (1 = Don't care, 0 = Exact match required)
    
    // --- Output Results ---
    output reg [`TOPPATTERNS-1:0] match_line, // Multi-bit bus indicating which patterns matched (Hits)
    output reg valid                          // Global valid flag, high when all cells finish evaluation
    );
    
    // Internal wires to collect the outputs from the individual CAM cells
    wire [`TOPPATTERNS-1:0] match_line_temp;
    wire [`TOPPATTERNS-1:0] valid_line_temp;
    
    // =========================================================================
    // 1. Parallel CAM Cell Instantiation
    // Generates an array of individual CAM cells based on the `TOPPATTERNS macro.
    // The target_pattern and xmask are broadcasted to all cells simultaneously.
    // =========================================================================
    genvar i;
    generate
        for (i=0; i<`TOPPATTERNS; i=i+1) begin: gen
            CAM CAM_inst(
                .CLK(CLK), 
                .set(set), 
                // Extract the specific 9-bit pattern for this cell from the packed bus
                .set_pattern(set_patterns[i*9 +: 9]), 
                .en(en), 
                .target_pattern(target_pattern), 
                .xmask(xmask), 
                .match(match_line_temp[i]), 
                .valid(valid_line_temp[i])
            );
        end
    endgenerate
    
    // =========================================================================
    // 2. Output Combinational Logic
    // =========================================================================
    always @(*) begin
        // Pass the parallel match results directly to the output bus
        match_line = match_line_temp;
        
        // The global valid signal is asserted only if ALL instantiated CAM cells
        // report a valid output (Logical AND reduction).
        valid = &(valid_line_temp);
    end
    
endmodule
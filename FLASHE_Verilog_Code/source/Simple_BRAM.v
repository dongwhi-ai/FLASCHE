`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : Simple_BRAM
// Description : A behavioral dual-port Block RAM model intended STRICTLY FOR 
//               SIMULATION PURPOSES. It acts as a shared memory interface 
//               between the AXI-Stream writer and the main DUT (TopModule). 
//               Note: The asynchronous read logic is implemented specifically 
//               for simulation speed and is not intended for hardware synthesis.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module Simple_BRAM #(
    parameter DEPTH = 96000                      // Default memory depth for image storage
)(
    input clk,
    
    // =========================================================
    // Port A: Write Interface (From bram_writer)
    // =========================================================
    input [16:0] addr_a,                         // Write address
    input [7:0]  din_a,                          // Data input to memory
    input        we_a,                           // Write enable flag
    
    // =========================================================
    // Port B: Read Interface (To TopModule)
    // =========================================================
    input [16:0] addr_b,                         // Read address
    output [7:0] dout_b                          // Data output from memory
);

    // Internal memory array declaration
    reg [7:0] mem [0:DEPTH-1];

    // =========================================================
    // Write Logic (Port A)
    // Synchronous write operation triggered on the positive clock edge.
    // =========================================================
    always @(posedge clk) begin
        if (we_a) begin
            mem[addr_a] <= din_a;
        end
    end

    // =========================================================
    // Read Logic (Port B)
    // Asynchronous read (continuous assignment) implemented to 
    // accelerate simulation performance. (Simulation Only)
    // =========================================================
    assign dout_b = mem[addr_b];

endmodule
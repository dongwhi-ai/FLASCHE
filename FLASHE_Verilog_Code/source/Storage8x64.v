`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : Storage8x64
// Description : A 64-byte dual-port memory buffer designed to store an 8x8 
//               image patch. It utilizes Vivado synthesis attributes to infer 
//               Block RAM (BRAM) for efficient area and power utilization.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module Storage8x64(
    input CLK,                  // System clock
    input [7:0] idata,          // 8-bit input data (pixel value)
    input write_en,             // Write enable signal
    input [5:0] write_addr,     // 6-bit write address (0 to 63)
    input read_en,              // Read enable signal
    input [5:0] read_addr,      // 6-bit read address (0 to 63)
    output reg [7:0] odata      // 8-bit output read data
    
    // --- Debug Interface (Optional) ---
    // output [8*64-1:0] all_data
    );
    
    // =========================================================
    // Memory Array Definition
    // The "ram_style" attribute forces the synthesis tool to 
    // map this 2D register array into a dedicated Block RAM.
    // =========================================================
    (* ram_style = "block" *)
    reg [7:0] memory [63:0];
    
    // =========================================================
    // Optional Debug Logic
    // Uncomment to expose the entire memory content as a flat vector 
    // for real-time monitoring via ILA or PS.
    // =========================================================
    /*
    genvar i;
    generate
      for (i = 0; i < 64; i = i + 1) begin
        assign all_data[i*8 +: 8] = memory[i];
      end
    endgenerate
    */
    
    // =========================================================
    // Sequential Logic: Dual-Port Memory Access
    // Supports simultaneous read and write operations.
    // =========================================================
    always @(posedge CLK) begin
        if (write_en == 1'b1) begin
            memory[write_addr] <= idata;
        end
        
        if (read_en == 1'b1) begin
            odata <= memory[read_addr];
        end
    end
    
endmodule
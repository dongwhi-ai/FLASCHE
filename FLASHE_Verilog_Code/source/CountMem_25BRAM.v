`timescale 1ns / 1ps
`include "defs.vh" 

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : CountMem_25BRAM
// Description : A memory module comprising 25 parallel Block RAMs to store 
//               pattern occurrence counts. It features hardware-accelerated 
//               memory clearing, saturating counters to prevent overflow, and 
//               pipelined read-modify-write paths.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module CountMem_25BRAM(
    input CLK,
    input RSTN,
    
    // --- Memory Initialization Signals ---
    input mem_clear,       // Trigger signal to start zeroing the memory
    output reg clear_done, // High when memory initialization is complete
    
    // --- Write Ports (Training Update) ---
    input  train_we,
    input  [3:0] w_filter_idx,
    input  [9*25-1:0] w_pattern_idxs, // Packed write addresses for 25 parallel patterns

    // --- Read Ports (Inference/Sorting Fetch) ---
    input  [3:0] r_filter_idx,
    input  [8:0] r_pattern_idx,       // Common read address shared across all 25 BRAMs

    // --- Output Ports ---
    output [8*25-1:0] counts_out_flat // Packed output of 25 parallel counts (8-bit each)
);

    // =========================================================================
    // 1. Hardware Memory Initialization (Clear) Logic
    // Sweeps through all memory addresses to reset counts to zero.
    // =========================================================================
    reg [12:0] clear_cnt;
    reg clearing_active;

    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            clear_cnt <= 13'd0;
            clearing_active <= 1'b0;
            clear_done <= 1'b0;
        end else begin
            if (mem_clear && !clearing_active) begin
                // Initiate the clearing process
                clearing_active <= 1'b1;
                clear_done <= 1'b0;
                clear_cnt <= 13'd0;
            end 
            else if (clearing_active) begin
                // Increment counter until the maximum address (6143) is reached
                if (clear_cnt < 13'd6143) begin
                    clear_cnt <= clear_cnt + 1;
                end else begin
                    clearing_active <= 1'b0;
                    clear_done <= 1'b1;
                end
            end else begin
                clear_done <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Pipeline Registers & Interconnects
    // =========================================================================
    reg        train_we_d1;
    reg [12:0] w_addr_pipe [0:24];
    
    reg [7:0]  mem_read_data [0:24]; // Raw read data directly from BRAM
    reg [7:0]  read_out_array [0:24]; // Registered output data

    always @(posedge CLK) begin
        // Block training updates while memory is being cleared
        train_we_d1 <= train_we && !clearing_active;
    end

    // Concatenate filter index and pattern index to form a 13-bit common read address
    wire [12:0] common_r_addr = {r_filter_idx, r_pattern_idx};

    // =========================================================================
    // 2. Instantiation of 25 Parallel BRAM Blocks
    // =========================================================================
    genvar i;
    generate
        for (i = 0; i < 25; i = i + 1) begin : MEM_BLK
            
            // [BRAM Declaration] 
            // The "ram_style" attribute enforces mapping to dedicated Block RAM.
            // Reset is strictly avoided here to allow optimal BRAM inference.
            (* ram_style = "block" *) reg [7:0] sub_mem [0:6143];
            
            // ----------------------------------------------------------------
            // [A] Combinational Routing: Address and Data MUX
            // ----------------------------------------------------------------
            wire [12:0] my_w_addr_in = {w_filter_idx, w_pattern_idxs[i*9 +: 9]};

            // 1. Write Enable: Active during clearing or pipelined training update
            wire b_we = clearing_active || train_we_d1;

            // 2. Write Address MUX: Route sweep counter during clear, else training address
            wire [12:0] b_w_addr = (clearing_active) ? clear_cnt : w_addr_pipe[i];

            // 3. Write Data MUX & Saturating Logic:
            // Prevents 8-bit overflow by capping the count value at 255 (8'hFF).
            wire [7:0]  next_val = (mem_read_data[i] == 8'hFF) ? 8'hFF : (mem_read_data[i] + 1);
            wire [7:0]  b_w_data = (clearing_active) ? 8'd0 : next_val;

            // 4. Read Address MUX:
            // During training, read the target address to modify it. Otherwise, read the common fetch address.
            wire [12:0] b_r_addr = (train_we) ? my_w_addr_in : common_r_addr;

            // ----------------------------------------------------------------
            // [B] Standard BRAM Inference (Synchronous Read/Write)
            // ----------------------------------------------------------------
            always @(posedge CLK) begin
                // Port A: Write Operation
                if (b_we) begin
                    sub_mem[b_w_addr] <= b_w_data;
                end
                
                // Port B: Read Operation
                mem_read_data[i] <= sub_mem[b_r_addr];
            end

            // ----------------------------------------------------------------
            // [C] Pipeline Registration
            // ----------------------------------------------------------------
            always @(posedge CLK) begin
                // Delay write address to align with BRAM read latency (Read-Modify-Write)
                if (train_we) begin
                    w_addr_pipe[i] <= my_w_addr_in;
                end
                
                // Register the BRAM output. 
                // Note: Asynchronous resets are intentionally omitted for optimal synthesis.
                if (!RSTN) begin
                    read_out_array[i] <= 8'd0;
                end else begin
                    read_out_array[i] <= mem_read_data[i];
                end
            end

        end
    endgenerate

    // =========================================================================
    // Output Formatting: Flatten the 2D array into a 1D packed bus
    // =========================================================================
    genvar k;
    generate
        for (k = 0; k < 25; k = k + 1) begin : OUT
            assign counts_out_flat[8*(k+1)-1 : 8*k] = read_out_array[k];
        end
    endgenerate

endmodule
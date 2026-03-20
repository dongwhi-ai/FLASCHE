`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : Top32Finder
// Description : A high-throughput Top-32 hardware sorter based on a systolic 
//               array architecture. It receives streamed counts and pattern 
//               indices, filters out adjacent duplicate inputs, and pushes 
//               the data through 32 pipelined compare-and-swap cells to 
//               maintain the top 32 highest counts in real-time.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module Top32Finder(
    input wire CLK,
    input wire RSTN,
    input wire soft_reset,                  // Synchronous reset for the sorter array
    input wire en,                          // Enable signal for data input
    input wire [7:0] new_count,             // Incoming pattern count (frequency)
    input wire [8:0] new_pattern_idx,       // Incoming pattern identifier (index)
    output wire [32*9-1:0] sorted_patterns, // Packed output of the Top-32 pattern indices
    output reg  ready                       // Ready flag indicating sorting completion
);
    localparam integer K = 32; // Defines the depth of the Top-K sorter

    // =========================================================================
    // 1. Input Processing & Control Registers
    // =========================================================================
    reg [7:0] r_new_count;
    reg [8:0] r_new_pattern_idx;
    reg       r_sort_en; 
    
    // Tracks the last processed index to filter out back-to-back duplicate requests
    reg [8:0] last_idx_reg;
    
    // Counter to push remaining data through the systolic array pipeline (Flush)
    reg [6:0] flush_cnt;    

    // Internal wires to read the state of each cell in the systolic array
    wire [8:0] cell_patterns [0:31]; 
    wire [7:0] cell_counts   [0:31];

    // =========================================================================
    // 2. Control Logic: Input Filtering & Pipeline Flush
    // =========================================================================
    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN || soft_reset) begin
            ready             <= 1'b1;
            r_new_count       <= 8'd0;
            r_new_pattern_idx <= 9'd0;
            r_sort_en         <= 1'b0;
            
            // Initialize with an out-of-bounds value (511) to ensure the very first 
            // valid pattern (even if it is index 0) is properly registered.
            last_idx_reg      <= 9'h1FF; 
            flush_cnt         <= 7'd0;
        end else begin
            // -----------------------------------------------------------------
            // CASE 1: Active Input State
            // Valid data is streaming in with a count greater than zero.
            // -----------------------------------------------------------------
            if (en && (new_count > 0)) begin 
                
                // Anti-Duplication Logic: 
                // Checks if the incoming pattern index is different from the last 
                // one processed to prevent redundant sorting operations.
                if (new_pattern_idx != last_idx_reg) begin
                    r_new_count       <= new_count;
                    r_new_pattern_idx <= new_pattern_idx;
                    last_idx_reg      <= new_pattern_idx; // Update duplicate tracker
                    r_sort_en         <= 1'b1;
                    
                    // Set flush counter to 32 (depth of the array). 
                    // This ensures all valid data is pushed to its final sorted 
                    // position even after the input stream stops.
                    flush_cnt         <= 7'd32; 
                    ready             <= 1'b1;  
                end else begin
                    // If the index matches last_idx_reg, it's a back-to-back duplicate.
                    // The sorter intentionally ignores it (drops the data).
                    // The flush_cnt is NOT reset, allowing pending data to move.
                    // We also intentionally do not clear last_idx_reg here.
                end
            end 
            // -----------------------------------------------------------------
            // CASE 2: Pipeline Flush State
            // Input stream stopped, but the array needs to keep shifting to 
            // complete the sorting of already ingested data.
            // -----------------------------------------------------------------
            else if (flush_cnt > 0) begin
                // Inject zeros (null data) to push existing data down the array
                r_new_count       <= 8'd0;
                r_new_pattern_idx <= 9'd0;
                r_sort_en         <= 1'b1;
                flush_cnt         <= flush_cnt - 1;
                ready             <= 1'b1; 
            end
            // -----------------------------------------------------------------
            // CASE 3: Idle State
            // Sorting pipeline is fully flushed and empty.
            // -----------------------------------------------------------------
            else begin
                r_sort_en         <= 1'b0;
                r_new_count       <= 8'd0;
                r_new_pattern_idx <= 9'd0;
                ready             <= 1'b1;
                
                // Note: last_idx_reg is intentionally preserved during IDLE to 
                // handle gaps in the input stream robustly without falsely 
                // resetting the anti-duplication logic.
            end
        end
    end

    // =========================================================================
    // 3. Systolic Array Instantiation (32 SortCells)
    // Instantiates K=32 identical sort cells cascaded in a linear pipeline.
    // =========================================================================
    wire [7:0] wires_count [0:32]; 
    wire [8:0] wires_pat   [0:32];
    
    // Connect the buffered input to the first cell of the array
    assign wires_count[0] = r_new_count;
    assign wires_pat[0]   = r_new_pattern_idx;

    genvar i;
    generate
        for (i = 0; i < K; i = i + 1) begin : CELL
            // Wires connecting the output of Cell[i] to the input of Cell[i+1]
            wire [7:0] next_count;
            wire [8:0] next_pat;
            
            assign wires_count[i+1] = next_count;
            assign wires_pat[i+1]   = next_pat;

            SortCell u_cell (
                .CLK(CLK), .RSTN(RSTN), .soft_reset(soft_reset),
                .en(r_sort_en), 
                .in_count(wires_count[i]), 
                .in_pattern_idx(wires_pat[i]),
                .out_count(next_count), 
                .out_pattern_idx(next_pat),
                .my_count(cell_counts[i]), 
                .my_pattern_idx(cell_patterns[i])
            );
        end
    endgenerate
    
    // =========================================================================
    // 4. Output Data Packing
    // Collects the sorted indices from all 32 cells and packs them into a single bus.
    // =========================================================================
    genvar k;
    generate
        for (k = 0; k < K; k = k + 1) begin : OUT_LOGIC
            // If the cell count is greater than 0, output its valid pattern index.
            // Otherwise, fall back to the highest ranked pattern (cell_patterns[0]) 
            // to fill empty slots safely, preventing invalid zero-index accesses.
            assign sorted_patterns[9*(k+1)-1 : 9*k] = 
                (cell_counts[k] > 8'd0) ? cell_patterns[k] : cell_patterns[0];
        end
    endgenerate

endmodule
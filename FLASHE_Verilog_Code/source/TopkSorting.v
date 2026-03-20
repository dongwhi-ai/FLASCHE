`timescale 1ns / 1ps

// ============================================================================
// Module: TopkSorting
// Description: A resource-efficient hardware Top-K sorter. It processes 16 
//              input counts sequentially over 16 clock cycles, utilizing a 
//              parallel-compare and shift-insert mechanism to maintain the 
//              top 12 values and their corresponding indices in descending order.
// Tool       : Xilinx Vivado 2024.2
// ============================================================================
module TopkSorting#(
    parameter integer CNT_NUM = 16,       // Total number of input elements
    parameter integer TOPK = 12,          // Number of top elements to retain (Top-K)
    parameter integer CNT_WIDTH = 16,     // Bit-width of the count values
    parameter integer IDX_WIDTH = 4       // Bit-width of the indices (0..15)
)(
    input  wire                     CLK,
    input  wire                     en,             // Enable signal to start sorting
    input  wire [CNT_NUM*CNT_WIDTH-1:0] counts,     // Packed input array of 16 count values
    
    output reg                      top_valid,      // High when the sorting is fully completed
    output reg  [TOPK*IDX_WIDTH-1:0] top_idx_packed,// Packed array of sorted Top-K indices
    output reg  [TOPK*CNT_WIDTH-1:0] top_val_packed // Packed array of sorted Top-K values
);

    // =========================================================
    // FSM State Definitions
    // =========================================================
    parameter IDLE    = 1'b0, 
              SORTING = 1'b1;
              
    reg state = IDLE;
    reg next_state;
    wire sorting_done;
    reg top_sorted_valid;
    
    // Register array to hold the incoming counts for sequential processing
    reg [CNT_WIDTH-1:0] counts_reg [CNT_NUM-1:0];
    
    // Internal Top-K Buffers (Sorted in descending order: index 0 is the maximum)
    reg [CNT_WIDTH-1:0] val_buf [TOPK-1:0];
    reg [IDX_WIDTH-1:0] idx_buf [TOPK-1:0];
    
    // Comparison flags for the shift-insert logic
    reg [TOPK-1+1:0] comp_flags;
    reg [CNT_WIDTH-1:0] cnt_target;

    // Counter to track the number of processed input elements (0 to 15)
    reg [7:0] processed_cnt;   
    assign sorting_done = (processed_cnt == CNT_NUM-1);

    integer i;
    integer k;
    
    // =========================================================
    // Sequential Logic: FSM & Datapath
    // =========================================================
    always @(posedge CLK) begin
        state <= next_state;
        
        // --- IDLE State: Initialization and Output Registration ---
        if (state==IDLE) begin
            if (en==1'b1) begin
                // Load incoming packed data into the internal register array
                for (i = 0; i < CNT_NUM; i = i + 1) begin
                    counts_reg[i] <= counts[i*CNT_WIDTH +: CNT_WIDTH];
                end
                // Clear the Top-K buffers
                for (i = 0; i < TOPK; i = i + 1) begin
                    val_buf[i] <= {CNT_WIDTH{1'b0}};
                    idx_buf[i] <= {IDX_WIDTH{1'b0}};
                end
                processed_cnt <= 0;
            end
            
            // Output the final packed results when sorting is fully finished
            if (top_sorted_valid) begin
                for (k = 0; k < TOPK; k = k + 1) begin
                    top_idx_packed[k*IDX_WIDTH +: IDX_WIDTH] <= idx_buf[k];
                    top_val_packed[k*CNT_WIDTH +: CNT_WIDTH] <= val_buf[k];
                end
                top_valid <= 1'b1;
            end
            else begin
                top_valid <= 1'b0;
            end
            top_sorted_valid <= 0;
        end
        
        // --- SORTING State: Shift-Insert Mechanism ---
        
        if (state==SORTING) begin
            for (i = 0; i < TOPK; i = i + 1) begin
                // Based on 2-bit slices of comp_flags, determine buffer action:
                if (comp_flags[i+:2]==2'b11) begin
                    // 11: Target is smaller than current and next buffer. Hold value.
                    val_buf[i] <= val_buf[i];
                    idx_buf[i] <= idx_buf[i];
                end
                else if (comp_flags[i+:2]==2'b01) begin
                    // 01: Target is larger than the current buffer but smaller than the previous. Insert target here.
                    val_buf[i] <= cnt_target;
                    idx_buf[i] <= processed_cnt[IDX_WIDTH-1:0];
                end
                else begin
                    // 00 / 10: Target is larger than previous buffers. Shift current value down.
                    val_buf[i] <= val_buf[i-1];
                    idx_buf[i] <= idx_buf[i-1];
                end
            end
            
            if (sorting_done) begin
                processed_cnt <= 0;
                top_sorted_valid <= 1'b1;
            end
            else begin
                processed_cnt <= processed_cnt + 1;
            end
        end
    end
    
    // =========================================================
    // Combinational Logic: Next State
    // =========================================================
    always @(*) begin
        case (state)
            IDLE:    next_state = en ? SORTING : IDLE;
            SORTING: next_state = sorting_done ? IDLE : SORTING;
            default: next_state = IDLE;
        endcase
    end
    
    // =========================================================
    // Combinational Logic: Parallel Comparison
    // Generates a multi-bit flag indicating where the current target 
    // count fits within the sorted buffer.
    // =========================================================
    always @(*) begin
        cnt_target = counts_reg[processed_cnt];
        
        // Compare target against all current Top-K values simultaneously.
        // A '1' indicates the target is smaller or equal to the buffer value.
        // The LSB is tied to 1 to act as a boundary condition for the shift-insert logic.
        comp_flags = {cnt_target <= val_buf[11], 
                      cnt_target <= val_buf[10], 
                      cnt_target <= val_buf[9], 
                      cnt_target <= val_buf[8], 
                      cnt_target <= val_buf[7], 
                      cnt_target <= val_buf[6], 
                      cnt_target <= val_buf[5], 
                      cnt_target <= val_buf[4], 
                      cnt_target <= val_buf[3], 
                      cnt_target <= val_buf[2], 
                      cnt_target <= val_buf[1], 
                      cnt_target <= val_buf[0], 
                      1'b1};
    end

endmodule
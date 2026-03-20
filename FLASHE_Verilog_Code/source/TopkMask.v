`timescale 1ns / 1ps

// ============================================================================
// Module: TopkMask
// Description: Receives an array of histogram counts, utilizes a sorting 
//              sub-module to identify the indices of the Top-K elements, 
//              and generates a 16-bit multi-hot mask indicating the selected 
//              top patterns.
// Tool       : Xilinx Vivado 2024.2
// ============================================================================
module TopkMask #(
    parameter integer CNT_NUM   = 16,   // Total number of input bins/candidates (16)
    parameter integer TOPK      = 12,   // Number of top elements to select (Top-K = 12)
    parameter integer CNT_WIDTH = 16,   // Bit-width of each count value
    parameter integer IDX_WIDTH = 4     // Bit-width of the index (0..15 requires 4 bits)
)(
    input  wire                     CLK,
    input  wire                     en,           // Enable signal to start sorting
    input  wire [CNT_NUM*CNT_WIDTH-1:0] counts,   // Packed array of 16 count values

    output reg                      top_valid,    // High when the top_mask output is ready
    output reg  [CNT_NUM-1:0]       top_mask      // 16-bit mask where Top-K indices are set to 1
);

    // =========================================================
    // TopkSorting Sub-module Interface
    // =========================================================
    wire                        sorting_valid;
    wire [TOPK*IDX_WIDTH-1:0]   top_idx_packed; // Packed array of the Top-K indices
    wire [TOPK*CNT_WIDTH-1:0]   top_val_packed; // Packed array of Top-K values (unused here)

    // Instantiate the hardware sorting module
    TopkSorting #(
        .CNT_NUM   (CNT_NUM),
        .TOPK      (TOPK),
        .CNT_WIDTH (CNT_WIDTH),
        .IDX_WIDTH (IDX_WIDTH)
    ) u_topk_sorting (
        .CLK            (CLK),
        .en             (en),
        .counts         (counts),
        .top_valid      (sorting_valid),
        .top_idx_packed (top_idx_packed),
        .top_val_packed (top_val_packed)
    );

    // =========================================================
    // Combinational Logic: Index to Mask Conversion
    // Decodes the packed Top-K indices into a multi-hot bitmask
    // =========================================================
    integer i;
    reg [CNT_NUM-1:0] mask_next;

    always @(*) begin
        mask_next = {CNT_NUM{1'b0}};  // Initialize mask to all zeros
        for (i = 0; i < TOPK; i = i + 1) begin
            // Extract the i-th index from the packed array and set the corresponding bit
            mask_next[
                top_idx_packed[i*IDX_WIDTH +: IDX_WIDTH]
            ] = 1'b1;
        end
    end

    // =========================================================
    // Sequential Logic: Output Registration
    // Updates the final mask output one cycle after sorting completes
    // =========================================================
    always @(posedge CLK) begin
        if (sorting_valid) begin
            top_mask  <= mask_next;
            top_valid <= 1'b1;
        end
        else begin
            top_valid <= 1'b0;
        end
    end

endmodule
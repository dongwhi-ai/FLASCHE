`timescale 1ns / 1ps

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : bram_writer
// Description : An AXI-Stream to BRAM bridge. It deserializes incoming 64-bit 
//               streamed data into 8-bit chunks for sequential BRAM writing. 
//               It features a pulse-stretched completion signal for robust 
//               system synchronization and automatic address reset upon TLAST 
//               detection, enabling continuous circular buffer operation.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module bram_writer #(
    parameter DEPTH = 96000
)(
    input aclk, 
    input aresetn,
    
    // --- AXI-Stream Slave Interface ---
    input [63:0] s_axis_tdata,
    input s_axis_tvalid,
    output s_axis_tready,
    input s_axis_tlast,
    
    // --- BRAM Master Interface ---
    output reg [16:0] bram_addr,
    output reg [7:0] bram_din,
    output reg bram_we,
    
    // --- System Control / Status ---
    output reg img_write_done
);
    
    // Internal state and data registers
    reg [2:0] byte_count;
    reg [63:0] data_reg;
    reg busy, is_last_transfer, first_signal_sent;
    
    // Counter to stretch the 'done' pulse width for slower clock domains or listeners
    reg [3:0] done_hold_count;

    // Ready to accept new data only when not busy deserializing the current 64-bit word
    assign s_axis_tready = !busy;

    // =========================================================================
    // Sequential Logic: FSM and Datapath
    // =========================================================================
    always @(posedge aclk) begin
        // ---------------------------------------------------------------------
        // 1. Asynchronous Reset
        // ---------------------------------------------------------------------
        if (!aresetn) begin
            byte_count        <= 0; 
            busy              <= 0; 
            bram_addr         <= 0; 
            bram_we           <= 0; 
            img_write_done    <= 0; 
            first_signal_sent <= 0;
            done_hold_count   <= 0;
        end else begin
            bram_we <= 0; // Default state: do not write
            
            // -----------------------------------------------------------------
            // 2. Pulse Stretching Logic for Completion Signal
            // Extends the 'img_write_done' flag for ~15 clock cycles to ensure
            // the downstream logic or CPU reliably captures the event.
            // -----------------------------------------------------------------
            if (done_hold_count > 0) begin
                img_write_done  <= 1'b1;
                done_hold_count <= done_hold_count - 1;
            end else begin
                img_write_done  <= 1'b0;
            end

            // -----------------------------------------------------------------
            // 3. AXI-Stream Data Reception
            // Latches the 64-bit data and TLAST flag when a valid handshake occurs.
            // -----------------------------------------------------------------
            if (!busy && s_axis_tvalid) begin
                data_reg         <= s_axis_tdata;
                busy             <= 1'b1; 
                byte_count       <= 0;
                is_last_transfer <= s_axis_tlast;
            end
            
            // -----------------------------------------------------------------
            // 4. Deserialization and BRAM Write Operations
            // -----------------------------------------------------------------
            if (busy) begin
                bram_we <= 1'b1;
                
                // Multiplex the 64-bit register into 8-bit sequential chunks
                case (byte_count)
                    0: bram_din <= data_reg[7:0];   1: bram_din <= data_reg[15:8];
                    2: bram_din <= data_reg[23:16]; 3: bram_din <= data_reg[31:24];
                    4: bram_din <= data_reg[39:32]; 5: bram_din <= data_reg[47:40];
                    6: bram_din <= data_reg[55:48]; 7: bram_din <= data_reg[63:56];
                endcase
                
                if (byte_count == 7) begin
                    // Finished writing all 8 bytes of the current 64-bit word
                    busy <= 0;
                    
                    // ---------------------------------------------------------
                    // Boundary Condition & Circular Buffer Reset
                    // If the TLAST flag was asserted for this transfer, reset 
                    // the BRAM address to 0 for the next incoming image frame.
                    // ---------------------------------------------------------
                    if (is_last_transfer) begin
                        bram_addr <= 0; // Wrap around to the start of the buffer
                        
                        // Trigger the pulse-stretched completion signal
                        if (!first_signal_sent) begin 
                             done_hold_count <= 4'd15;
                             // Uncomment the line below if the signal should only trigger once per reset
                             // first_signal_sent <= 1'b1; 
                        end
                    end else begin
                        // Proceed to the next byte address
                        bram_addr <= bram_addr + 1;
                    end
                    
                end else begin
                    // Increment byte counter and memory address for the next chunk
                    byte_count <= byte_count + 1;
                    bram_addr  <= bram_addr + 1;
                end
            end
        end
    end
endmodule
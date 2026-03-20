`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : TrainingModule
// Description : FSM-driven main controller for the on-chip training phase. 
//               It orchestrates pattern extraction, occurrence counting via BRAM, 
//               parallel Top-K sorting, and finally updates the inference memory.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module TrainingModule #(
    parameter integer TOP_K         = 32,
    parameter integer NUM_FILTERS   = 12,
    parameter integer IMAGE_WIDTH   = 49,
    parameter integer PAT_STAB_CYCLES = 3,  // Cycles to stabilize pattern extraction
    parameter integer MEM_WR_CYCLES   = 2,  // Cycles for memory write operations
    parameter integer MEM_RD_CYCLES   = 2   // Cycles for memory read operations
)(
    input  wire                     CLK,
    input  wire                     RSTN,
    input  wire                     train_pulse,      // Trigger to start training
    input  wire                     last_image_flag,  // High if current image is the last in batch
    input  wire [3:0]               target_class,     // Ground truth class label
    input  wire [`IMAGEARR2-1:0]    image00, image01, image02, image03,
    input  wire [`IMAGEARR2-1:0]    image04, image05, image06, image07,
    input  wire [`IMAGEARR2-1:0]    image08, image09, image10, image11,
    
    // --- Inference Memory Update Interface ---
    output wire                     infer_we,
    output wire [8:0]               infer_addr,
    output wire [71:0]              infer_data,
    output wire [5:0]               infer_sel,
    
    // --- Status Flags ---
    output reg                      sample_done,      // High when a single sample is processed
    output wire                     done              // High when the entire training phase completes
);

    // =========================================================
    // FSM State Definitions
    // =========================================================
    reg [3:0] state;
    localparam S_IDLE        = 0, 
               S_TRAIN_PRE   = 1,  // Prepare pattern extraction
               S_TRAIN_REQ   = 2,  // Request memory write for counts
               S_TRAIN_WAIT  = 3,  // Wait for write completion
               S_SORT_INIT   = 4,  // Initialize parallel sorters
               S_SORT_ADDR   = 5,  // Issue read address to BRAM
               S_SORT_WAIT   = 6,  // Wait for read data & sort
               S_UPDATE_REQ  = 7,  // Request inference memory update
               S_UPDATE_WAIT = 8,  // Wait for update completion
               S_DONE        = 9,  // Training done
               S_MEM_CLEAR   = 10, // Clear count BRAM for next class
               S_BOOT        = 11; // Boot sequence: Initial memory clear for CLASS0

    // Control Signals & Counters
    reg mem_we, mem_clear, sorter_reset, sorter_en, start_update, is_last_batch;
    reg [3:0] loop_filter_cnt;
    reg [8:0] scan_pat_idx;
    reg [8:0] scan_pat_idx_delayed; // Delayed index to align with BRAM read latency
    reg [7:0] wait_cnt;

    // Sorter Status Buses
    wire [24:0] sorter_ready_bus;
    wire sorter_ready = &sorter_ready_bus; // High when all 25 sorters are ready
    wire [TOP_K*9-1:0] sorted_pos_data [0:24];
    wire [25*TOP_K*9-1:0] all_sorted_bus;
    wire [8*25-1:0] counts_parallel;
    
    // =========================================================
    // Stage 1: Image Input Packing & Pattern Extraction
    // =========================================================
    wire [`IMAGEARR2-1:0] img_array [0:11];
    assign img_array[0]=image00; assign img_array[1]=image01; assign img_array[2]=image02; assign img_array[3]=image03;
    assign img_array[4]=image04; assign img_array[5]=image05; assign img_array[6]=image06; assign img_array[7]=image07;
    assign img_array[8]=image08; assign img_array[9]=image09; assign img_array[10]=image10; assign img_array[11]=image11;

    wire [9*25-1:0] extracted_patterns [0:11];
    genvar gi;
    generate
        for (gi = 0; gi < 12; gi = gi + 1) begin : PE
            PatternExtract PE_I (
                .CLK(CLK), .en(1'b1), .image_flat(img_array[gi]), 
                .patterns(extracted_patterns[gi]), .valid()
            );
        end
    endgenerate

    // Delay scan index to match BRAM read latency during sorting phase
    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            scan_pat_idx_delayed <= 9'd0;
        end else if (state == S_IDLE) begin
            scan_pat_idx_delayed <= 9'd0; 
        end else begin
            scan_pat_idx_delayed <= scan_pat_idx;
        end
    end

    // =========================================================
    // Stage 2: Memory Operations (Counting Occurrences)
    // =========================================================
    wire [9*25-1:0] current_train_pat = extracted_patterns[loop_filter_cnt];
    
    // Routing logic for write pattern addresses: 
    // Uses scan_pat_idx during MEM_CLEAR, extracted patterns during TRAIN, else zero.
    wire [9*25-1:0] final_w_pattern_idxs = (state == S_MEM_CLEAR) ? {25{scan_pat_idx}} : 
                                           (mem_we)               ? current_train_pat  : {9*25{1'b0}};

    wire clear_done_sig;

    CountMem_25BRAM MEM (
        .CLK(CLK), 
        .RSTN(RSTN),             
        .mem_clear(mem_clear), 
        .clear_done(clear_done_sig), // High when BRAM zeroing is complete
        .train_we(mem_we),
        .w_filter_idx(loop_filter_cnt), 
        .w_pattern_idxs(final_w_pattern_idxs),
        .r_filter_idx(loop_filter_cnt), 
        .r_pattern_idx(scan_pat_idx),
        .counts_out_flat(counts_parallel)
    );

    // =========================================================
    // Stage 3: Parallel Top-K Sorting
    // Instantiates 25 parallel sorters to identify the most frequent patterns
    // =========================================================
    genvar sj;
    generate
        for (sj = 0; sj < 25; sj = sj + 1) begin : SORTERS
            Top32Finder SORTER_I (
                .CLK(CLK), .RSTN(RSTN), .soft_reset(sorter_reset), .en(sorter_en),
                .new_count(counts_parallel[8*(sj+1)-1 : 8*sj]),
                .new_pattern_idx(scan_pat_idx_delayed), 
                .ready(sorter_ready_bus[sj]),           
                .sorted_patterns(sorted_pos_data[sj])
            );
        end
    endgenerate

    // Pack the 25 individual sorter outputs into a single flat bus
    genvar pk;
    generate
        for (pk = 0; pk < 25; pk = pk + 1) begin : PACK
            assign all_sorted_bus[(pk+1)*TOP_K*9-1 : pk*TOP_K*9] = sorted_pos_data[pk];
        end
    endgenerate

    // =========================================================
    // Stage 4: Inference Memory Update
    // Translates sorted results into format required by the inference engine
    // =========================================================
    wire update_done_sig;
    InferenceUpdateModule UPDATER (
        .CLK(CLK), .RSTN(RSTN), .start_update(start_update), .target_class(target_class),
        .current_filter_id(loop_filter_cnt), .sorted_data_flat(all_sorted_bus),
        .infer_we(infer_we), .infer_addr(infer_addr), .infer_data(infer_data),
        .infer_sel(infer_sel), .update_done(update_done_sig)
    );

    assign done = (state == S_DONE);
    
    // =========================================================
    // Main Sequential Logic: FSM State Transitions & Control
    // =========================================================
    always @(posedge CLK or negedge RSTN) begin
        if (!RSTN) begin
            // Boot sequence starts with S_BOOT instead of S_IDLE to clear BRAM
            state <= S_BOOT; 
            loop_filter_cnt <= 0; scan_pat_idx <= 0;
            mem_we <= 0; mem_clear <= 0; sorter_reset <= 0; sorter_en <= 0;
            start_update <= 0; is_last_batch <= 0; sample_done <= 0; wait_cnt <= 0;
        end else begin
            // Default assignments to prevent latch inference and unintended glitches
            mem_we       <= 1'b0; 
            mem_clear    <= 1'b0; 
            sorter_reset <= 1'b0; 
            sorter_en    <= 1'b0;
            start_update <= 1'b0; 
            sample_done  <= 1'b0;

            case (state)
                S_BOOT: begin
                    mem_clear <= 1'b1;    // Trigger memory initialization (1 cycle pulse)
                    state <= S_MEM_CLEAR; // Move to wait state
                end

                S_IDLE: begin
                    // Wait for training command
                    if (train_pulse) begin
                        is_last_batch   <= last_image_flag;
                        loop_filter_cnt <= 0;
                        wait_cnt        <= PAT_STAB_CYCLES[7:0];
                        state           <= S_TRAIN_PRE;
                    end
                end

                S_TRAIN_PRE: begin
                    if (wait_cnt == 0) state <= S_TRAIN_REQ;
                    else wait_cnt <= wait_cnt - 1;
                end

                S_TRAIN_REQ: begin
                    mem_we    <= 1'b1; // Trigger memory write for occurrence count
                    mem_clear <= 1'b0; // Ensure clear signal is off
                    wait_cnt  <= MEM_WR_CYCLES[7:0];
                    state     <= S_TRAIN_WAIT;
                end

                S_TRAIN_WAIT: begin
                    // Turn off mem_we (pulse generation complete)
                    if (wait_cnt == 0) begin
                        if (loop_filter_cnt == (NUM_FILTERS-1)) begin
                            if (is_last_batch) begin
                                loop_filter_cnt <= 0; 
                                state           <= S_SORT_INIT; 
                            end else begin
                                sample_done <= 1; 
                                state       <= S_IDLE; 
                            end
                        end else begin
                            loop_filter_cnt <= loop_filter_cnt + 1;
                            wait_cnt        <= PAT_STAB_CYCLES[7:0]; 
                            state           <= S_TRAIN_PRE;
                        end
                    end else wait_cnt <= wait_cnt - 1;
                end

                S_SORT_INIT: begin
                    sorter_reset <= 1'b1; // Reset sorter arrays for a new class/filter
                    scan_pat_idx <= 0; 
                    state        <= S_SORT_ADDR;
                end

                S_SORT_ADDR: begin
                    wait_cnt <= MEM_RD_CYCLES[7:0]; 
                    state    <= S_SORT_WAIT;
                end

                S_SORT_WAIT: begin
                    if (wait_cnt == 0 && sorter_ready) begin
                        sorter_en <= 1; // Trigger sorter to process fetched data
                        if (scan_pat_idx == 9'd511) begin
                            state <= S_UPDATE_REQ;
                        end else begin
                            scan_pat_idx <= scan_pat_idx + 1'b1;
                            state        <= S_SORT_ADDR;
                        end
                    end else if (wait_cnt != 0) begin
                        wait_cnt <= wait_cnt - 1'b1;
                    end
                end

                S_UPDATE_REQ: begin
                    start_update <= 1; 
                    state        <= S_UPDATE_WAIT;
                end

                S_UPDATE_WAIT: begin
                    if (update_done_sig) begin
                        if (loop_filter_cnt == 11) begin
                            loop_filter_cnt <= 0;
                            scan_pat_idx    <= 0;
                            
                            if (target_class == 4'd9) begin 
                                state <= S_DONE;      
                            end else begin
                                // Clear BRAM before transitioning to next class
                                mem_clear <= 1'b1;     // Pulse clear signal
                                state     <= S_MEM_CLEAR; 
                            end
                        end else begin
                            loop_filter_cnt <= loop_filter_cnt + 1;
                            state <= S_SORT_INIT;
                        end
                    end
                end
                
                S_MEM_CLEAR: begin
                    // Wait for BRAM initialization to finish
                    // Note: mem_clear was pulsed for 1 cycle, so it remains 0 here
                    if (clear_done_sig) begin
                        state <= S_DONE; // Proceed to done state
                    end else begin
                        state <= S_MEM_CLEAR; // Wait
                    end
                end

                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
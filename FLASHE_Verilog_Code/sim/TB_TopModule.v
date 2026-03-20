`timescale 1ns / 1ps
`include "defs.vh"

// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : TB_TopModule
// Description : Top-level testbench for the AI accelerator system. It verifies 
//               the complete operational pipeline including AXI-Stream image 
//               loading via BRAM writer, the Edge Counting (sampling) phase, 
//               the Training phase across 10 classes, and the final Inference 
//               phase. Includes automated file logging for internal state debug.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================
module TB_TopModule;

    // =========================================================
    // Clock and Reset Generators
    // =========================================================
    reg CLK, RSTN;
    
    initial begin
        CLK = 0;
        RSTN = 0;
    end
    
    always #5 CLK = ~CLK; // 100MHz Clock
    
    // =========================================================
    // Signal Declarations
    // =========================================================
    // Input Image Buffer
    reg [`PIXELBITS-1:0] pixel_inputs [0:10240-1];
    
    // Control Signals for TopModule
    reg edge_count_en; 
    reg edge_count_stop; 
    reg [3:0] class; 
    reg train_en; 
    reg train_last;
    reg sort_en; 
    reg prediction_en; 
    
    // Outputs from TopModule
    wire [3:0] prediction; 
    wire prediction_valid; 
    reg writer_rstn;
    
    // AXI-Stream & BRAM Interface Signals
    reg [63:0] s_axis_tdata;
    reg s_axis_tvalid, s_axis_tlast;
    wire s_axis_tready;
    wire [16:0] wr_addr, rd_addr;
    wire [7:0]  wr_data, rd_data;
    wire wr_en, img_trigger;
    
    // =========================================================
    // Module Instantiations
    // =========================================================
    
    // 1. AXI-Stream to BRAM Writer
    bram_writer writer_inst (
        .aclk(CLK), 
        .aresetn(writer_rstn),
        .s_axis_tdata(s_axis_tdata), 
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready), 
        .s_axis_tlast(s_axis_tlast),
        .bram_addr(wr_addr), 
        .bram_din(wr_data), 
        .bram_we(wr_en),
        .img_write_done(img_trigger)
    );
    
    // 2. Shared Dual-Port Memory (Simple BRAM)
    Simple_BRAM shared_mem (
        .clk(CLK),
        .addr_a(wr_addr), .din_a(wr_data), .we_a(wr_en), 
        .addr_b(rd_addr), .dout_b(rd_data)                
    );
    
    // 3. Main Device Under Test (DUT)
    TopModule TopModule_inst (
        .CLK(CLK), 
        .RSTN(RSTN), 
        .img_write_done(img_trigger), 
        .rd_addr(rd_addr),            
        .pixel(rd_data), 
        .train_en(train_en), 
        .train_last(train_last), 
        .sort_en(sort_en), 
        .edge_count_en(edge_count_en), 
        .edge_count_stop(edge_count_stop), 
        .class(class), 
        .prediction_en(prediction_en), 
        .prediction(prediction), 
        .prediction_valid(prediction_valid)
    );
    
    integer i, j, k;
    
    // =========================================================
    // [Task] AXI-Stream Data Transmission
    // Normalizes pixel order (LSB First Packing) and manages handshake
    // =========================================================
    task send_axis;
        input integer offset;      
        input integer num_pixels; 
        integer idx;
        begin
            // 1. Reset Writer (Clock synchronized)
            writer_rstn = 0; 
            repeat(5) @(posedge CLK); 
            
            // Release reset and allow hardware to initialize
            writer_rstn = 1; 
            repeat(10) @(posedge CLK); 

            // 2. Data Transmission Loop
            for (idx = offset; idx < offset + num_pixels; idx = idx + 8) begin
                // Wait until the slave is ready to receive data
                wait(s_axis_tready); 
                
                @(posedge CLK); 
                s_axis_tvalid = 1;
                
                // Pack 8 pixels into 64-bit data {p7...p0} -> p0 is mapped to LSB
                s_axis_tdata = {
                    pixel_inputs[idx+7], pixel_inputs[idx+6], 
                    pixel_inputs[idx+5], pixel_inputs[idx+4], 
                    pixel_inputs[idx+3], pixel_inputs[idx+2], 
                    pixel_inputs[idx+1], pixel_inputs[idx+0]  
                };
                
                // Assert tlast signal for the final 64-bit packet
                s_axis_tlast = (idx >= offset + num_pixels - 8); 
                
                // [Critical] Verify AXI-Stream Handshake (tvalid && tready)
                while (s_axis_tready == 0) begin
                    @(posedge CLK);
                end
                
                @(posedge CLK); 
                s_axis_tvalid = 0;
                s_axis_tlast = 0;
            end
            
            // 3. Write Completion Delay
            // Provide buffer time for BRAM to securely store the transmitted data
            repeat(20) @(posedge CLK); 
        end
    endtask
    
    // =========================================================
    // Main Simulation Scenario
    // =========================================================
    initial begin
        // Initialize Control Signals
        edge_count_en = 0; 
        edge_count_stop = 0; 
        class = 0; 
        train_en = 0;
        train_last = 0;
        sort_en = 0;
        prediction_en = 0;
        writer_rstn = 0;
        
        #100;
        RSTN = 1;
        writer_rstn = 1;
        #10;
        
        // ---------------------------------------------------------
        // Phase 1. Edge Count Phase (Sampling)
        // Accurately loads images per class and applies repeat(64) + #1 timing
        // ---------------------------------------------------------
        $display("=== Starting Edge Count Phase (Image-by-Image) ===");
        
        // --- Class 0 (149 Images) ---
        $display("Edge Counting Class 0...");
        class = 0;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class0_train_images.txt"}, pixel_inputs);
        edge_count_en = 0; 
        for (j = 0; j < 149; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20; 
            wait(CLK == 0); edge_count_en = 1'b1; 
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0; 
            #50; 
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 1 (152 Images) ---
        $display("Edge Counting Class 1...");
        class = 1;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class1_train_images.txt"}, pixel_inputs);
        edge_count_en = 0; 
        for (j = 0; j < 152; j = j + 1) begin
            send_axis(j * 64, 64);
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 2 (148 Images) ---
        $display("Edge Counting Class 2...");
        class = 2;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class2_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 148; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 3 (153 Images) ---
        $display("Edge Counting Class 3...");
        class = 3;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class3_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 153; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 4 (151 Images) ---
        $display("Edge Counting Class 4...");
        class = 4;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class4_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 151; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 5 (152 Images) ---
        $display("Edge Counting Class 5...");
        class = 5;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class5_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 152; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 6 (151 Images) ---
        $display("Edge Counting Class 6...");
        class = 6;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class6_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 151; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 7 (149 Images) ---
        $display("Edge Counting Class 7...");
        class = 7;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class7_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 149; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 8 (145 Images) ---
        $display("Edge Counting Class 8...");
        class = 8;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class8_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 145; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        
        // --- Class 9 (150 Images) ---
        $display("Edge Counting Class 9...");
        class = 9;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class9_train_images.txt"}, pixel_inputs);
        edge_count_en = 0;
        for (j = 0; j < 150; j = j + 1) begin
            send_axis(j * 64, 64); 
            #20;
            wait(CLK == 0); edge_count_en = 1'b1;
            repeat(64) @(posedge CLK); #1; edge_count_en = 1'b0;
            #50;
        end
        edge_count_stop = 1; #10; edge_count_stop = 0; #1000;
        

        // ---------------------------------------------------------
        // Phase 2. START TRAINING PHASE (All 10 Classes)
        // ---------------------------------------------------------
        $display("=== Starting Training Phase ===");

        // --- Class 0 (149 Images) ---
        $display("Training Class 0... (149 images)");
        class = 0;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class0_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 148; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0); 
            #100;
        end
        train_en = 1'b1;
        send_axis(148 * 64, 64);
        train_last = 1'b1; 
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0); 
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 1 (152 Images) ---
        $display("Training Class 1... (152 images)");
        class = 1;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class1_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 151; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(151 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0); 
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 2 (148 Images) ---
        $display("Training Class 2... (148 images)");
        class = 2;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class2_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 147; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(147 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 3 (153 Images) ---
        $display("Training Class 3... (153 images)");
        class = 3;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class3_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 152; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(152 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 4 (151 Images) ---
        $display("Training Class 4... (151 images)");
        class = 4;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class4_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 150; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(150 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 5 (152 Images) ---
        $display("Training Class 5... (152 images)");
        class = 5;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class5_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 151; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(151 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 6 (151 Images) ---
        $display("Training Class 6... (151 images)");
        class = 6;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class6_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 150; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(150 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 7 (149 Images) ---
        $display("Training Class 7... (149 images)");
        class = 7;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class7_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 148; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(148 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 8 (145 Images) ---
        $display("Training Class 8... (145 images)");
        class = 8;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class8_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 144; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(144 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        // --- Class 9 (150 Images) ---
        $display("Training Class 9... (150 images)");
        class = 9;
        for (k = 0; k < 10240; k = k + 1) pixel_inputs[k] = 0;
        $readmemh({`IMAGE_BASE, "class9_train_images.txt"}, pixel_inputs);
        for (j = 0; j < 149; j = j + 1) begin 
            train_en = 1'b1;
            send_axis(j * 64, 64); 
            wait(TopModule_inst.state == 4'h0);
            #100;
        end
        train_en = 1'b1;
        send_axis(149 * 64, 64); 
        train_last = 1'b1;
        #20;
        wait(TopModule_inst.state != 4'h0); 
        wait(TopModule_inst.state == 4'h0);
        train_en = 1'b0; train_last = 1'b0; #10000;

        $display("All Training Classes Completed.");
        
        // ---------------------------------------------------------
        // Phase 3. START INFERENCE PHASE (10 Test Images)
        // ---------------------------------------------------------
        $display("=== Starting Inference Phase ===");

        // --- Image 0 ---
        $display("Testing Image 0...");
        $readmemh({`IMAGE_BASE, "pred_0image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);                   
        wait(TopModule_inst.state != 4'h0); 
        prediction_en = 1'b0;               
        wait(TopModule_inst.prediction_valid); 
        #1000;

        // --- Image 1 ---
        $display("Testing Image 1...");
        $readmemh({`IMAGE_BASE, "pred_1image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 2 ---
        $display("Testing Image 2...");
        $readmemh({`IMAGE_BASE, "pred_2image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 3 ---
        $display("Testing Image 3...");
        $readmemh({`IMAGE_BASE, "pred_3image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 4 ---
        $display("Testing Image 4...");
        $readmemh({`IMAGE_BASE, "pred_4image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 5 ---
        $display("Testing Image 5...");
        $readmemh({`IMAGE_BASE, "pred_5image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 6 ---
        $display("Testing Image 6...");
        $readmemh({`IMAGE_BASE, "pred_6image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 7 ---
        $display("Testing Image 7...");
        $readmemh({`IMAGE_BASE, "pred_7image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 8 ---
        $display("Testing Image 8...");
        $readmemh({`IMAGE_BASE, "pred_8image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;
        
        // --- Image 9 ---
        $display("Testing Image 9...");
        $readmemh({`IMAGE_BASE, "pred_9image.txt"}, pixel_inputs);
        prediction_en = 1'b1;
        send_axis(0, 64);
        wait(TopModule_inst.state != 4'h0);
        prediction_en = 1'b0;
        wait(TopModule_inst.prediction_valid);
        #1000;

        $display("All Inference Tests Completed.");
        $finish;
    end
    
    // =================================================================
    // [Debug] EdgeCounter Input Logger
    // Records the actual values observed by the EdgeCounter inside TopModule.
    // =================================================================
    integer f_log;
    initial begin
        // Use "golden_log.txt" for Golden Model TB
        f_log = $fopen("bram_log.txt", "w"); 
        
        $fwrite(f_log, "Time       | Class | Counter | En | Pixel  | Addr (BRAM only)\n");
        $fwrite(f_log, "-----------+-------+---------+----+--------+-----------------\n");
    end

    // Capture data on every posedge of the clock.
    // A #1 delay is added to ensure stable values are captured right after the clock edge.
    always @(posedge CLK) begin
        #1; 
        // Record only during the Edge Count phase to prevent massive log files
        if (edge_count_en || TopModule_inst.state == 4'h5) begin // 4'h5 = EDGETOPCOUNT
            $fwrite(f_log, "%t |   %d   |  %4d   | %b  |   %h   |      %h\n", 
                    $time, 
                    class,
                    TopModule_inst.counter,              // Internal counter
                    TopModule_inst.EdgeCounter_inst.en,  // EdgeCounter enable flag
                    TopModule_inst.pixel,                // Pixel input to EdgeCounter
                    TopModule_inst.rd_addr               // (Reference) BRAM address
            );
        end
    end
    
endmodule
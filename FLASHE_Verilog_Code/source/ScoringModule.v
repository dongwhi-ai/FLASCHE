`timescale 1ns / 1ps
`include "defs.vh"
// ============================================================================
// Copyright (c) 2026 Kyung Hee University
// Author      : Integrated Circuits (IC) Lab
// Module      : ScoringModule
// Description : The core inference engine that evaluates input image features 
//               against stored Top-K patterns using CAM array emulation. 
//               It orchestrates a multi-stage pipeline: BRAM Fetch -> Pattern 
//               Extraction -> CAM Match Detection -> Score Mapping -> Accumulation.
// Tool        : Xilinx Vivado 2024.2
// ============================================================================

// BRAM read must be synchronous
// en -> bram read $ pattern extract -> cam set & target pattern -> match detect -> score mapping -> score accumulation



module ScoringModule(
    input CLK, 
    input RSTN, 
    // --- Input Feature Maps (12 Parallel Channels) ---
    input [`IMAGEARR2-1:0] image00, 
    input [`IMAGEARR2-1:0] image01, 
    input [`IMAGEARR2-1:0] image02, 
    input [`IMAGEARR2-1:0] image03, 
    input [`IMAGEARR2-1:0] image04, 
    input [`IMAGEARR2-1:0] image05, 
    input [`IMAGEARR2-1:0] image06, 
    input [`IMAGEARR2-1:0] image07, 
    input [`IMAGEARR2-1:0] image08, 
    input [`IMAGEARR2-1:0] image09, 
    input [`IMAGEARR2-1:0] image10, 
    input [`IMAGEARR2-1:0] image11, 
    input [`IMAGEARR2-1:0] xmask, 
    input en, 
    // --- Inference Memory Update Interface (from Training Module) ---
    input  wire        infer_we, 
    input  wire [8:0]  infer_addr, 
    input  wire [71:0] infer_data, 
    input  wire [5:0]  infer_sel, 
    // --- Output Results ---
    output reg [3:0] class, // The class currently being evaluated
    output reg [16*10-1:0] scores, // Packed array of accumulated scores for 10 classes
    output reg score_valid  // High when the final scores are valid
    );
    // =========================================================================
    // 1. Inference Weight Memory (32 Block RAMs)
    // Stores the trained Top-K patterns. Explicitly mapped to BRAM for efficiency.
    // =========================================================================
    (* ram_style = "block" *) reg [71:0] TOPPatterns1 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns2 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns3 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns4 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns5 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns6 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns7 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns8 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns9 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns10 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns11 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns12 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns13 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns14 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns15 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns16 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns17 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns18 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns19 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns20 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns21 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns22 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns23 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns24 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns25 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns26 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns27 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns28 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns29 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns30 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns31 [511:0];
    (* ram_style = "block" *) reg [71:0] TOPPatterns32 [511:0];
    // BRAM Read Data Registers
    reg [71:0] top1s;
    reg [71:0] top2s;
    reg [71:0] top3s;
    reg [71:0] top4s;
    reg [71:0] top5s;
    reg [71:0] top6s;
    reg [71:0] top7s;
    reg [71:0] top8s;
    reg [71:0] top9s;
    reg [71:0] top10s;
    reg [71:0] top11s;
    reg [71:0] top12s;
    reg [71:0] top13s;
    reg [71:0] top14s;
    reg [71:0] top15s;
    reg [71:0] top16s;
    reg [71:0] top17s;
    reg [71:0] top18s;
    reg [71:0] top19s;
    reg [71:0] top20s;
    reg [71:0] top21s;
    reg [71:0] top22s;
    reg [71:0] top23s;
    reg [71:0] top24s;
    reg [71:0] top25s;
    reg [71:0] top26s;
    reg [71:0] top27s;
    reg [71:0] top28s;
    reg [71:0] top29s;
    reg [71:0] top30s;
    reg [71:0] top31s;
    reg [71:0] top32s;
    // Group the 32 read outputs into 8 parallel sets for the CAM arrays
    wire [9*32-1:0] set_patterns [7:0];
    assign set_patterns[0] = {top32s[8:0], top31s[8:0], top30s[8:0], top29s[8:0], top28s[8:0], top27s[8:0], top26s[8:0], top25s[8:0], top24s[8:0], top23s[8:0], top22s[8:0], top21s[8:0], top20s[8:0], top19s[8:0], top18s[8:0], top17s[8:0], top16s[8:0], top15s[8:0], top14s[8:0], top13s[8:0], top12s[8:0], top11s[8:0], top10s[8:0], top9s[8:0], top8s[8:0], top7s[8:0], top6s[8:0], top5s[8:0], top4s[8:0], top3s[8:0], top2s[8:0], top1s[8:0]};
    assign set_patterns[1] = {top32s[17:9], top31s[17:9], top30s[17:9], top29s[17:9], top28s[17:9], top27s[17:9], top26s[17:9], top25s[17:9], top24s[17:9], top23s[17:9], top22s[17:9], top21s[17:9], top20s[17:9], top19s[17:9], top18s[17:9], top17s[17:9], top16s[17:9], top15s[17:9], top14s[17:9], top13s[17:9], top12s[17:9], top11s[17:9], top10s[17:9], top9s[17:9], top8s[17:9], top7s[17:9], top6s[17:9], top5s[17:9], top4s[17:9], top3s[17:9], top2s[17:9], top1s[17:9]};
    assign set_patterns[2] = {top32s[26:18], top31s[26:18], top30s[26:18], top29s[26:18], top28s[26:18], top27s[26:18], top26s[26:18], top25s[26:18], top24s[26:18], top23s[26:18], top22s[26:18], top21s[26:18], top20s[26:18], top19s[26:18], top18s[26:18], top17s[26:18], top16s[26:18], top15s[26:18], top14s[26:18], top13s[26:18], top12s[26:18], top11s[26:18], top10s[26:18], top9s[26:18], top8s[26:18], top7s[26:18], top6s[26:18], top5s[26:18], top4s[26:18], top3s[26:18], top2s[26:18], top1s[26:18]};
    assign set_patterns[3] = {top32s[35:27], top31s[35:27], top30s[35:27], top29s[35:27], top28s[35:27], top27s[35:27], top26s[35:27], top25s[35:27], top24s[35:27], top23s[35:27], top22s[35:27], top21s[35:27], top20s[35:27], top19s[35:27], top18s[35:27], top17s[35:27], top16s[35:27], top15s[35:27], top14s[35:27], top13s[35:27], top12s[35:27], top11s[35:27], top10s[35:27], top9s[35:27], top8s[35:27], top7s[35:27], top6s[35:27], top5s[35:27], top4s[35:27], top3s[35:27], top2s[35:27], top1s[35:27]};
    assign set_patterns[4] = {top32s[44:36], top31s[44:36], top30s[44:36], top29s[44:36], top28s[44:36], top27s[44:36], top26s[44:36], top25s[44:36], top24s[44:36], top23s[44:36], top22s[44:36], top21s[44:36], top20s[44:36], top19s[44:36], top18s[44:36], top17s[44:36], top16s[44:36], top15s[44:36], top14s[44:36], top13s[44:36], top12s[44:36], top11s[44:36], top10s[44:36], top9s[44:36], top8s[44:36], top7s[44:36], top6s[44:36], top5s[44:36], top4s[44:36], top3s[44:36], top2s[44:36], top1s[44:36]};
    assign set_patterns[5] = {top32s[53:45], top31s[53:45], top30s[53:45], top29s[53:45], top28s[53:45], top27s[53:45], top26s[53:45], top25s[53:45], top24s[53:45], top23s[53:45], top22s[53:45], top21s[53:45], top20s[53:45], top19s[53:45], top18s[53:45], top17s[53:45], top16s[53:45], top15s[53:45], top14s[53:45], top13s[53:45], top12s[53:45], top11s[53:45], top10s[53:45], top9s[53:45], top8s[53:45], top7s[53:45], top6s[53:45], top5s[53:45], top4s[53:45], top3s[53:45], top2s[53:45], top1s[53:45]};
    assign set_patterns[6] = {top32s[62:54], top31s[62:54], top30s[62:54], top29s[62:54], top28s[62:54], top27s[62:54], top26s[62:54], top25s[62:54], top24s[62:54], top23s[62:54], top22s[62:54], top21s[62:54], top20s[62:54], top19s[62:54], top18s[62:54], top17s[62:54], top16s[62:54], top15s[62:54], top14s[62:54], top13s[62:54], top12s[62:54], top11s[62:54], top10s[62:54], top9s[62:54], top8s[62:54], top7s[62:54], top6s[62:54], top5s[62:54], top4s[62:54], top3s[62:54], top2s[62:54], top1s[62:54]};
    assign set_patterns[7] = {top32s[71:63], top31s[71:63], top30s[71:63], top29s[71:63], top28s[71:63], top27s[71:63], top26s[71:63], top25s[71:63], top24s[71:63], top23s[71:63], top22s[71:63], top21s[71:63], top20s[71:63], top19s[71:63], top18s[71:63], top17s[71:63], top16s[71:63], top15s[71:63], top14s[71:63], top13s[71:63], top12s[71:63], top11s[71:63], top10s[71:63], top9s[71:63], top8s[71:63], top7s[71:63], top6s[71:63], top5s[71:63], top4s[71:63], top3s[71:63], top2s[71:63], top1s[71:63]};
    // =========================================================================
    // 2. FSM & Pipeline Control Counters
    // =========================================================================
    reg busy;
    reg [2:0] waiting;
    
    wire [8:0] bram_addr;
    reg [3:0] bram_counter;
    reg [3:0] bram_counter_1d;
    reg [3:0] bram_counter_2d;
    reg [3:0] filter_counter;
    reg [3:0] class_counter;
    reg [3:0] class_counter_1d;
    reg [3:0] class_counter_2d;
    reg [3:0] class_counter_3d;
    reg [3:0] class_counter_4d;
    reg [3:0] class_counter_5d;
    assign bram_addr = class_counter*48 + filter_counter*4 + bram_counter;
    
    // PatternExtract
    reg pattern_ext_en;
    reg [`IMAGEARR2-1:0] target_image;
    wire [9*25-1:0] patterns;
    wire pattern_ext_valid;
    PatternExtract PatternExtract_inst_0(
        .CLK(CLK), 
        .en(pattern_ext_en), 
        .image_flat(target_image), 
        .patterns(patterns), 
        .valid(pattern_ext_valid)
    );
    wire [9*25-1:0] xmasks;
    wire xmask_ext_valid;
    PatternExtract PatternExtract_inst_1(
        .CLK(CLK), 
        .en(pattern_ext_en), 
        .image_flat(xmask), 
        .patterns(xmasks), 
        .valid(xmask_ext_valid)
    );
    
    // CAMArray
    reg set_en;
    reg [7:0] cam_ens;
    reg [8:0] target_patterns [7:0];
    reg [8:0] target_xmasks [7:0];
    wire [`TOPPATTERNS-1:0] match_lines [7:0];
    wire [7:0] match_line_valids;
    genvar j;
    generate
        for (j=0; j<8; j=j+1) begin: gen
            CAMArray CAMArray_inst(
                .CLK(CLK), 
                .set_patterns(set_patterns[j]), 
                .set(set_en), 
                .en(cam_ens[j]), 
                .target_pattern(target_patterns[j]), 
                .xmask(target_xmasks[j]),
                .match_line(match_lines[j]), 
                .valid(match_line_valids[j])
            );
        end
    endgenerate
    
    // OneHotScoreMapper
    reg [7:0] mapping_ens;
    wire [15:0] acc_temp;
    wire mapping_valid;
    OneHotScoreMapperArray OneHotScoreMapperArray_inst(
        .CLK(CLK), 
        .ens(mapping_ens), 
        .match_lines({match_lines[7], match_lines[6], match_lines[5], match_lines[4], match_lines[3], match_lines[2], match_lines[1], match_lines[0]}), 
        .score(acc_temp), 
        .valid(mapping_valid)
    );
    
    // ScoreAccArray
    reg [9:0] acc_ens;
    wire [9:0] clrs;
    assign clrs = {10{score_valid}};
    wire [16*10-1:0] scores_temp;
    wire score_valid_temp;
    ScoreAccArray ScoreAccArray_inst(
        .CLK(CLK), 
        .RSTN(RSTN), 
        .ens(acc_ens), 
        .clrs(clrs), 
        .acc(acc_temp), 
        .scores(scores_temp), 
        .valid(score_valid_temp)
    );
    
    always @(posedge CLK) begin
        if (!RSTN) begin
          // 필요하면 초기화
        end else if (infer_we) begin
            case (infer_sel[4:0])
                5'd0:  TOPPatterns1 [infer_addr] <= infer_data;
                5'd1:  TOPPatterns2 [infer_addr] <= infer_data;
                5'd2:  TOPPatterns3 [infer_addr] <= infer_data;
                5'd3:  TOPPatterns4 [infer_addr] <= infer_data;
                5'd4:  TOPPatterns5 [infer_addr] <= infer_data;
                5'd5:  TOPPatterns6 [infer_addr] <= infer_data;
                5'd6:  TOPPatterns7 [infer_addr] <= infer_data;
                5'd7:  TOPPatterns8 [infer_addr] <= infer_data;
                5'd8:  TOPPatterns9 [infer_addr] <= infer_data;
                5'd9:  TOPPatterns10[infer_addr] <= infer_data;
                5'd10: TOPPatterns11[infer_addr] <= infer_data;
                5'd11: TOPPatterns12[infer_addr] <= infer_data;
                5'd12: TOPPatterns13[infer_addr] <= infer_data;
                5'd13: TOPPatterns14[infer_addr] <= infer_data;
                5'd14: TOPPatterns15[infer_addr] <= infer_data;
                5'd15: TOPPatterns16[infer_addr] <= infer_data;
                5'd16: TOPPatterns17[infer_addr] <= infer_data;
                5'd17: TOPPatterns18[infer_addr] <= infer_data;
                5'd18: TOPPatterns19[infer_addr] <= infer_data;
                5'd19: TOPPatterns20[infer_addr] <= infer_data;
                5'd20: TOPPatterns21[infer_addr] <= infer_data;
                5'd21: TOPPatterns22[infer_addr] <= infer_data;
                5'd22: TOPPatterns23[infer_addr] <= infer_data;
                5'd23: TOPPatterns24[infer_addr] <= infer_data;
                5'd24: TOPPatterns25[infer_addr] <= infer_data;
                5'd25: TOPPatterns26[infer_addr] <= infer_data;
                5'd26: TOPPatterns27[infer_addr] <= infer_data;
                5'd27: TOPPatterns28[infer_addr] <= infer_data;
                5'd28: TOPPatterns29[infer_addr] <= infer_data;
                5'd29: TOPPatterns30[infer_addr] <= infer_data;
                5'd30: TOPPatterns31[infer_addr] <= infer_data;
                5'd31: TOPPatterns32[infer_addr] <= infer_data;
                default: ;
            endcase
        end
    end
    
    integer i;
    always @(posedge CLK) begin
        if (!RSTN) begin
            busy <= 0;
            waiting <= 0;
            bram_counter <= 0;
            filter_counter <= 0;
            class_counter <= 0;
        end
        else begin
            cam_ens[7:1] <= {7{(set_en==1'b1) && !(bram_counter_2d==4'd2)}};
            cam_ens[0] <= set_en;
            bram_counter_1d <= bram_counter;
            bram_counter_2d <= bram_counter_1d;
            class_counter_1d <= class_counter;
            class_counter_2d <= class_counter_1d;
            class_counter_3d <= class_counter_2d;
            class_counter_4d <= class_counter_3d;
            class_counter_5d <= class_counter_4d;
            waiting[1] <= waiting[0];
            waiting[2] <= waiting[1];
            if (en==1'b1 && busy==0 && waiting==3'b000) begin
                busy <= 1'b1;
            end
            if (busy==1'b1) begin
                top1s <= TOPPatterns1[bram_addr];
                top2s <= TOPPatterns2[bram_addr];
                top3s <= TOPPatterns3[bram_addr];
                top4s <= TOPPatterns4[bram_addr];
                top5s <= TOPPatterns5[bram_addr];
                top6s <= TOPPatterns6[bram_addr];
                top7s <= TOPPatterns7[bram_addr];
                top8s <= TOPPatterns8[bram_addr];
                top9s <= TOPPatterns9[bram_addr];
                top10s <= TOPPatterns10[bram_addr];
                top11s <= TOPPatterns11[bram_addr];
                top12s <= TOPPatterns12[bram_addr];
                top13s <= TOPPatterns13[bram_addr];
                top14s <= TOPPatterns14[bram_addr];
                top15s <= TOPPatterns15[bram_addr];
                top16s <= TOPPatterns16[bram_addr];
                top17s <= TOPPatterns17[bram_addr];
                top18s <= TOPPatterns18[bram_addr];
                top19s <= TOPPatterns19[bram_addr];
                top20s <= TOPPatterns20[bram_addr];
                top21s <= TOPPatterns21[bram_addr];
                top22s <= TOPPatterns22[bram_addr];
                top23s <= TOPPatterns23[bram_addr];
                top24s <= TOPPatterns24[bram_addr];
                top25s <= TOPPatterns25[bram_addr];
                top26s <= TOPPatterns26[bram_addr];
                top27s <= TOPPatterns27[bram_addr];
                top28s <= TOPPatterns28[bram_addr];
                top29s <= TOPPatterns29[bram_addr];
                top30s <= TOPPatterns30[bram_addr];
                top31s <= TOPPatterns31[bram_addr];
                top32s <= TOPPatterns32[bram_addr];
                
                if (bram_counter==4'd3) begin
                    bram_counter <= 0;
                    if (filter_counter==4'd11) begin
                        filter_counter <= 0;
                        if (class_counter==4'd9) begin
                            class_counter <= 0;
                            waiting <= 1'b1;
                            busy <= 0;
                        end
                        else begin
                            class_counter <= class_counter + 1;
                        end
                    end
                    else begin
                        filter_counter <= filter_counter + 1;
                    end
                end
                else begin
                    bram_counter <= bram_counter + 1;
                end
            end
            if (waiting[0]==1'b1 && score_valid==1'b1) begin
                waiting[0] <= 0;
            end
            case (bram_counter_1d)
                8'd0: begin
                    for (i=0; i<8; i=i+1) begin
                        target_patterns[i] <= patterns[i*9 +: 9];
                        target_xmasks[i] <= xmasks[i*9 +: 9];
                    end
                end
                8'd1: begin
                    for (i=0; i<8; i=i+1) begin
                        target_patterns[i] <= patterns[(i+8)*9 +: 9];
                        target_xmasks[i] <= xmasks[(i+8)*9 +: 9];
                    end
                end
                8'd2: begin
                    for (i=0; i<8; i=i+1) begin
                        target_patterns[i] <= patterns[(i+16)*9 +: 9];
                        target_xmasks[i] <= xmasks[(i+16)*9 +: 9];
                    end
                end
                8'd3: begin
                    for (i=0; i<8; i=i+1) begin
                        if (i==0) begin
                            target_patterns[i] <= patterns[224:216];
                            target_xmasks[i] <= xmasks[224:216];
                        end
                        else begin
                            target_patterns[i] <= 0;
                            target_xmasks[i] <= 0;
                        end
                    end
                end
                default: begin
                    
                end
            endcase
        end
    end
    
    always @(*) begin
        class = class_counter;
        pattern_ext_en = busy;
        set_en = pattern_ext_valid;
        mapping_ens = match_line_valids;
        case (filter_counter)
            8'd0: begin
                target_image = image00;
            end
            8'd1: begin
                target_image = image01;
            end
            8'd2: begin
                target_image = image02;
            end
            8'd3: begin
                target_image = image03;
            end
            8'd4: begin
                target_image = image04;
            end
            8'd5: begin
                target_image = image05;
            end
            8'd6: begin
                target_image = image06;
            end
            8'd7: begin
                target_image = image07;
            end
            8'd8: begin
                target_image = image08;
            end
            8'd9: begin
                target_image = image09;
            end
            8'd10: begin
                target_image = image10;
            end
            8'd11: begin
                target_image = image11;
            end
            default: begin
                target_image = 0;
            end
        endcase
        for (i=0; i<10; i=i+1) begin
            acc_ens[i] = mapping_valid && (i==class_counter_5d);
        end
        
        scores = scores_temp;
        score_valid = (waiting[0]==1'b1) && (score_valid_temp==0);
    end
    
endmodule

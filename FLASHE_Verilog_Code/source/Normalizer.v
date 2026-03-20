`timescale 1ns / 1ps
`include "defs.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/07/04 13:27:54
// Design Name: 
// Module Name: Normalizer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Normalizer(
    input CLK, 
    input en, 
    input [`PIXELBITS-1:0] idata, 
    input [`PIXELBITS+4-1:0] old_max, 
    input [`PIXELBITS-1:0] new_max, 
    output reg valid, 
    output reg [`PIXELBITS-1:0] odata
    ); 
    
    reg [`PIXELBITS-1:0] odata_temp;
    reg [`PIXELBITS+4-1:0] mul_result;
    reg mul_valid;
    
    always @(posedge CLK) begin
        mul_valid <= en;
        valid <= mul_valid;
        if (en==1'b1) begin
            mul_result <= idata * new_max;
        end
        if (valid==1'b1) begin
            odata <= mul_result / old_max;
        end
        else begin
            odata <= 0;
        end
    end
    
endmodule

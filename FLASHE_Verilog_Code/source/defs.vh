/*
test_param = {'top_2quant_threshold': 12, 
                'score_2quant_threshold': 8, 
                'eval_threshold_gap0': 1, 
                'eval_threshold_gap1': 9, 
                'eval_threshold_gap2': 11, 
                'eval_threshold_gap3': 1, 
                'eval_threshold_gap4': 9, 
                'eval_threshold_gap5': 2, 
                'score_val0': 82, 
                'score_val1': 52, 
                'score_val2': 43, 
                'score_val3': 9, 
                'score_val4': 69, 
                'score_val5': 46}
*/
/*
pattern_idx = np.where(ctcam_sorted[:, 0] == pattern)[0]
if (ctcam_sorted[pattern_idx, 1]==0): # countê°? 0?´ë©? rankê°? ë²”ìœ„ ?•ˆ?´?–´?„ 0? ì²˜ë¦¬ ?˜?„ë¡? ?•¨.
    continue
else:
    rank = pattern_idx + 1
    for (threshold, score_val) in zip(thresholds, score_vals):
        if rank < threshold:
            score += score_val
            score_detailed[row, col] = score_val
            break
*/

`define COUNT_BASE "C:/Users/skrhi/verilog_test_data/counts/"
`define IMAGE_BASE "C:/Users/skrhi/verilog_test_data/images/"

`define CLASSES 10 // ScoringModule.v
`define PIXELBITS 8

`define IMAGELEN1 8
`define IMAGEARR1 64
`define IMAGELEN2 7
`define IMAGEARR2 49

`define BITHRTOP 12
`define BITHRSCR 8
`define TRITHR1 3
`define TRITHR2 12
`define RANKTHR1 1
`define RANKTHR2 10
`define RANKTHR3 21
`define RANKTHR4 22
`define RANKTHR5 31
`define RANKTHR6 33
`define SCORE1 82
`define SCORE2 52
`define SCORE3 43
`define SCORE4 9
`define SCORE5 69
`define SCORE6 46

`define PRELAYERMAX 16
`define NEWMAX 15

`define STORAGELEN 8

`define USEMATCH 0

`define COUNTBITS 9
`define PARALLELCOMP 32
`define ADDRWIDTH 5
`define STRIDE 1
`define CTCAMLEN 5
`define CTCAMARR 25 // ScoringModule.v

`define DATAWIDTH 72 // ScoringModule.v
`define TOPPATTERNS 32 // ScoringModule.v, ... // if you changed TOPPATTERNS, must change "case statement in always @(*) in ScoreAcc.v".
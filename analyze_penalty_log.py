import re

def analyze_penalty_log(penalty_log):
    """
    Given a penalty log, returns:
    - whether the destination was reached in each trial
    - number of penalties for each trial
    - number of penalties with clear intersections for each trial
    """
    with open(penalty_log, 'r') as f:
        string = f.read()
        
    trials = [patterns[0] for patterns in
              re.findall(r'(Trial \d{1,2}.*?)(Simulator|$)', 
                         string, re.DOTALL)]

    dest_reached = [len(re.findall('reached destination', trial))
                    for trial in trials]

    penalties = [len(re.findall('penalty!', trial)) for trial in trials]

    penalties_no_traffic = [len(re.findall('oncoming: None, left: None',
                                           trial))
                            for trial in trials]
                                
    print "number of destinations reached: ", sum(dest_reached)
    print "last unreached destination: trial", max([i for i in range(len(dest_reached)) 
                                                    if not dest_reached[i]])
    print "total number of penalties: ", sum(penalties)
    print "penalties during first 50 trials:", sum(penalties[:50])
    print "total number of penalties on clear intersections:", sum(penalties_no_traffic)
    print "last penalty on clear intersection:", max([i for i in range(len(penalties_no_traffic)) 
                                                     if penalties_no_traffic[i]])
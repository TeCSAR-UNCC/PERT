import numpy as np

mapping_ucla2ntu = {
    #    0:  Pick up with one hand	5. Pick up
    #    1:  Pick up with two hand
    #
    #    2:  Drop Trash	            4. Drop
    #
    #    3:  Walk around	        58. walking towards each other
    #                               59. walking apart from each other.
    #
    #    4:  Sit Down	            7. sitting down.
    #
    #    5:  Stand Up	            8. standing up (from sitting position)
    #
    #    6:  Donning	            13. wear jacket.
    #                               15. wear a shoe.
    #                               17. wear on glasses.
    #                               19. put on a hat/cap.
    #
    #    7:  Doffing	            14. take off jacket
    #                               16. take off a shoe
    #                               18. take off glasses
    #                               20. take off a hat/cap
    #
    #    8:  Throw	                6. throw
    #
    #    9:  Carry	                113. carry something with other person
    #
    0: [5],
    1: [5],
    2: [4],
    3: [58, 59],
    4: [7],
    5: [8],
    6: [13, 15, 17, 19],
    7: [14, 16, 18, 20],
    8: [6],
    9: [113],
}


def top1_accuracy(preds, targets, mapping):
    correct = 0
    for i, pred in enumerate(preds):
        if targets[i] in mapping.get(pred, []):
            correct += 1
    return correct / len(preds)


def top5_accuracy(preds, targets, mapping):
    # Assuming preds is a list of lists where each sublist is the top 5 predictions for that entry
    correct = 0
    for i, top_5 in enumerate(preds):
        valid_targets = set()
        for pred in top_5:
            valid_targets.update(mapping.get(pred, []))
        if targets[i] in valid_targets:
            correct += 1
    return correct / len(preds)

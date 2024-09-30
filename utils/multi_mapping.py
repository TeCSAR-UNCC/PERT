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

mapping_ucla2ucla = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
}


def calculate_accuracies(preds, targets, mapping):
    top1_correct = 0
    top5_correct = 0
    total_items = 0

    for pred_array, target_array in zip(preds, targets):
        for i in range(len(target_array)):
            target = target_array[i]
            top5_preds = pred_array[i]

            # Coressponding NTU labels:
            cntu_mapping = mapping.get(target, [])

            # Check top-1 accuracy
            if top5_preds[0] in cntu_mapping:
                top1_correct += 1

            # Check top-5 accuracy
            if np.isin(cntu_mapping, top5_preds).any():
                top5_correct += 1

        total_items += len(target_array)

    top1_accuracy = top1_correct / total_items
    top5_accuracy = top5_correct / total_items

    return top1_accuracy, top5_accuracy

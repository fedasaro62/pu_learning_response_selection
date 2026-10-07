from FineTuning.BERT_finetuning.data import _truncate_seq_pair


def test_truncate_leaves_short_pairs_alone():
    a, b = [1, 2, 3], [4, 5]

    _truncate_seq_pair(a, b, 10)

    assert (a, b) == ([1, 2, 3], [4, 5])


def test_truncate_drops_from_the_front_of_the_context():
    # context is history, so the oldest tokens go first
    a, b = [1, 2, 3, 4, 5], [6]

    _truncate_seq_pair(a, b, 4)

    assert a == [3, 4, 5] and b == [6]


def test_truncate_drops_from_the_end_of_the_response():
    a, b = [1], [2, 3, 4, 5, 6]

    _truncate_seq_pair(a, b, 4)

    assert a == [1] and b == [2, 3, 4]


def test_truncate_alternates_to_shorten_the_longer_sequence():
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]

    _truncate_seq_pair(a, b, 4)

    assert len(a) + len(b) == 4

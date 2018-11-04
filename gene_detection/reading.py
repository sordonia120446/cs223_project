def _spot_special_codon(seq):
    pass


def identify_open_reading_frames(seq):
    """
    Given a sequence s over the alphabet SDNA = (a, c, g, t).
    Is said that a subsequence t of SDNA is an open reading frame (ORF),
    if:
        t length is a multiple of three, i.e., t consists of a series of codons.
        t begins with the ATG start codon.
        t ends by one of the stop codons TAA, TAG or TGA.
    """
    pass

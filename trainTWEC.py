from twec.twec import TWEC


def train_twec_and_save(
        year
        , data_augmented_path
        , model_path
        , aligner=None
        , overwrite=True
        , size=300
        , siter=10
        , diter=10
        , workers=4
        , opath="model"
):
    """
    year: str
    data_augmented_path: the path to the data_augmented file that holds all data_augmented
    size: Number of dimensions. Default is 100.
    sg: Neural architecture of Word2vec. Default is CBOW (). If 1, Skip-gram is employed.
    siter: Number of static iterations (epochs). Default is 5.
    diter: Number of dynamic iterations (epochs). Default is 5.
    ns: Number of negative sampling examples. Default is 10, min is 1.
    window: Size of the context window (left and right). Default is 5 (5 left + 5 right).
    alpha: Initial learning rate. Default is 0.025.
    min_count: Min frequency for words over the entire corpus. Default is 5.
    workers: Number of worker threads. Default is 2.
    test: Folder name of the diachronic corpus files for testing.
    opath: Name of the desired output folder. Default is model.
    rtype: None (model saved to certain model path)
    """

    if not aligner:
        aligner = TWEC(size, siter, diter, workers)
        aligner.train_compass(data_augmented_path + "compass.txt", overwrite)
    model = aligner.train_slice(data_augmented_path + year + ".txt", save=True)
    model.save(model_path + year + ".model")


def train_period(start, end, compare, path="/Users/horus/PycharmProjects/SemanticProj/"):
    if compare == "engineering":
        path += "compare_engi/"
    elif compare == "material science":
        path += "compare_matsci/"
    DATAPATH = path + "data_augmented/"
    MODELPATH = path + "model/"

    for yr in range(start, end):
        train_twec_and_save(str(yr), data_augmented_path=DATAPATH, model_path=MODELPATH)

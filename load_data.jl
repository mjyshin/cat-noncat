using Images
using DataFrames
using CSV


function load_data()
    """
    LOAD DATASET
    Inputs:     Images
    Processes:  Resize and grayscale
    Outputs:    Train and test data and labels
    """
    ntrain = 209; ntest = 50; height = 64; width = 64

    train_set_x_orig = zeros(Int,ntrain,height,width,3)
    for i = 1:ntrain
        img = load("catvnoncat/train_set_x_orig/$(i).png")
        train_set_x_orig[i,:,:,:] = Int.(permutedims(float64.(rawview(channelview(img))), [2,3,1]))
    end
    df = CSV.read("catvnoncat/train_set_y.csv", header=false, nullable=false)
    train_set_y = convert(Array, df)

    test_set_x_orig = zeros(Int,ntest,height,width,3)
    for i = 1:ntest
        img = load("catvnoncat/test_set_x_orig/$(i).png")
        test_set_x_orig[i,:,:,:] = Int.(permutedims(float64.(rawview(channelview(img))), [2,3,1]))
    end
    df = CSV.read("catvnoncat/test_set_y.csv", header=false, nullable=false)
    test_set_y = convert(Array, df)

    df = CSV.read("catvnoncat/classes.csv", header=false, nullable=false)
    classes = convert(Array, df)

    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes
end

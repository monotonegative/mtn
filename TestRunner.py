import ImageRecognizerTest as test



# Entry point, main script 



# build model and verify it on the same dataset
test.verfiy_trained_model_on_test_dataset()


# tests on real images
test.load_random_webimage_and_recognize() 

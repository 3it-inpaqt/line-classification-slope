from utils.settings import settings

if __name__ == '__main__':

    run_type = settings.model_type  # fetch the type of NN to use
    for _ in range(settings.run_number):
        if run_type == 'FF':
            from models.run_regression import main
            main()

        elif run_type == 'CNN':
            from models.run_cnn import main
            main()

        elif run_type == 'EDGE-DETECT':
            from models.edge_detection import main
            main()

    # image_set_test, angles_test = create_image_set(n, N, 0.9, aa=True)
    # fig, axes = create_multiplots(image_set_test, angles_test, number_sample=n)
    # plt.show()


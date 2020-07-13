from billys.util import get_data_home, get_filename, read_dump, read_image
from billys.ocr.ocr import ocr_data
from billys.text.preprocessing import preprocess, make_nlp

if __name__ == "__main__":

    dump = read_dump(get_filename('trained_classifier.pkl'))
    clf = dump['classifier']
    train, text = dump['data']

    target_names = train['target_names'].unique().tolist()
    target_names.sort()

    new_image = read_image(get_filename('photo_2020-07-12_14-11-50.jpg'))

    ocr_dict = ocr_data(new_image)
    text = preprocess(text=" ".join(ocr_dict['text']), nlp=make_nlp())

    predicted = clf.predict([text])
    print([target_names[pred] for pred in predicted])

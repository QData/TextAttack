import textattack.datasets as datasets

def test_ag_news_load():

    # Expected
    expected_label = 0
    expected_text = (
        "Thirst, Fear and Bribes on Desert Escape from Africa AGADEZ, Niger (Reuters) - Customs officers in this dusty "
        "Saharan town turned a blind eye as yet another creaking truck piled with grain, smuggled cigarettes and dozens "
        "of migrants heading for Europe rumbled off into the desert."
    )

    # Actual
    data = datasets.classification.AGNews().__next__()
    print(data[1])
    print(expected_text)
    actual_label = data[1]
    actual_text = data[0]

    # Test
    assert expected_label == actual_label
    assert expected_text == actual_text


def test_imdb_load():

    # Expected 
    expected_label = 0
    expected_text = (
        "I bought this film on DVD so I could get an episode of Mystery Science Theater 3000. " 
        "Thankfully, Mike, Crow, and Tom Servo are watchable, because the film itself is not. " 
        "Although there is a plot, a story one can follow, and a few actors that can act, there isn't " 
        "anything else. The movie was so boring, I have firmly confirmed that I will never watch it again " 
        "without Tom, Crow and Mike. As summarized above, however, it was better than the film featured in "
        "the MST3K episode that preceded it; Mitchell."
    )   

    # Actual
    data = datasets.classification.IMDBSentiment().__next__()
    actual_label = data[1]
    actual_text = data[0]

    # Test
    assert expected_label == actual_label
    assert expected_text == actual_text


def test_movie_review_load():

    # Expected
    expected_label = 0
    expected_text = "possibly the most irresponsible picture ever released by a major film studio ."
    
    # Actual
    data = datasets.classification.MovieReviewSentiment().__next__()
    actual_label = data[1]
    actual_text = data[0]

    assert expected_label == actual_label
    assert expected_text == actual_text


def test_yelp_load():

    # Expected
    expected_label = 1
    expected_text = (
        "Mmmm nothing better than a late night Filly-B run! Had three shredded beef tacos. "
        "High Quality with all of the ingredients that you love is what you get here! I "
        "know they are bad for me but they taste so good..nom..nom..nom.."
    )

    # Actual
    data = datasets.classification.YelpSentiment().__next__()
    actual_label = data[1]
    actual_text = data[0]

    assert expected_label == actual_label
    assert expected_text == actual_text


def test_MNLI_load():

    # Expected 
    expected_label = 0
    expected_text = (
        "In Temple Bar , the bookshop at the Gallery of Photography carries a large selection "
        "of photographic publications , and the Flying Pig is a secondhand bookshop .>>>>"
        "There is a bookshop at the gallery ."
    )

    # Actual
    data = datasets.entailment.MNLI().__next__()
    actual_label = data[1]
    actual_text = data[0]

    assert expected_label == actual_label
    assert expected_text == actual_text







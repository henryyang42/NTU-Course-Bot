# from .models import Review
# from haystack import indexes


# class ReviewIndex(indexes.SearchIndex, indexes.Indexable):
#     text = indexes.NgramField(document=True, use_template=True)

#     title = indexes.CharField(model_attr='title')
#     content = indexes.CharField(model_attr='content')

#     def get_model(self):
#         return Review

#define DEFINE_ITERATOR(name, iter_t, value_t)                   \
    class name : public IteratorAdaptor<name, iter_t, value_t> { \
     public:                                                     \
        explicit name(iter_t start) : IteratorAdaptor(start){};  \
        value_t& dereference() const { return *base()->second; } \
    };

using EdgeMap = std::map<EDGE_ID_T, std::unique_ptr<Edge>>;
using BlobMap = std::map<BLOB_ID_T, std::unique_ptr<Blob>>;

DEFINE_ITERATOR(EdgeIt, EdgeMap::iterator, Edge);
DEFINE_ITERATOR(EdgeConstIt, EdgeMap::const_iterator, Edge);

DEFINE_ITERATOR(BlobIt, BlobMap::iterator, Blob);
DEFINE_ITERATOR(BlobConstIt, BlobMap::const_iterator, Blob);

#undef DEFINE_ITERATOR

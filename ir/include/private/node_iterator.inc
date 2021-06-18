#define DEFINE_ITERATOR(name, value_t, iter_t)                   \
    class name : public IteratorAdaptor<name, iter_t, value_t> { \
     public:                                                     \
        explicit name(iter_t start) : IteratorAdaptor(start){};  \
                                                                 \
        value_t& dereference() const { return *base(); }         \
    };

DEFINE_ITERATOR(NodeIt, Node, NodesList::iterator);
DEFINE_ITERATOR(NodeConstIt, const Node, NodesList::const_iterator);
DEFINE_ITERATOR(NodeReverseIt, Node, NodesList::reverse_iterator);
DEFINE_ITERATOR(NodeConstReverseIt, const Node, NodesList::const_reverse_iterator);

#undef DEFINE_ITERATOR

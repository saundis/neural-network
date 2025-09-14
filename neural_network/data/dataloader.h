#pragma once
#include "./datasets.h"
#include <memory>
#include <utility>
#include <vector>

// Used for loading data into batches and shuffling
class DataLoader
{
private:
    Dataset* m_dataset{};
    int m_batch_size{};
    // Used to determine order when iterating through dataset
    std::vector<int> m_indices{};

public:
    DataLoader(Dataset* dataset, int batch_size, bool shuffle = true);

    // Class to be able to iterate over 
    class Iterator
    {
    private:
        DataLoader* m_dataloader{};
        int m_index{};

    public:
        Iterator(DataLoader *dataloader, int index);
        // Index to next batch
        void operator++() { m_index += m_dataloader->m_batch_size; };
        // Returns vector of (label, image) in current batch
        std::vector<std::pair<int, std::shared_ptr<Tensor>>> operator*();
        // Check if you have indexed over all batches
        bool operator!=(const Iterator &other) { return m_index != other.m_index; };
    };

    // Returns iterator contains start iterator
    Iterator begin() { return Iterator(this, 0); };
    // Returns iterator representing when you need to stop iterating
    Iterator end() { return Iterator(this, m_dataset->get_length()); };

    std::size_t batch_size() const { return m_batch_size; };
    std::size_t n_samples() const { return m_dataset->get_length(); };
    std::size_t n_batches() const;
};

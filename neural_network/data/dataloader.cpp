#include "./dataloader.h"
#include "./datasets.h"
#include <algorithm>
#include <numeric>
#include <random>

DataLoader::DataLoader(Dataset* dataset, int batch_size, bool shuffle)
    : m_dataset(dataset), m_batch_size(batch_size)
{
    // Shuffles the data
    m_indices.resize(m_dataset->get_length());
    std::iota(m_indices.begin(), m_indices.end(), 0);
    if (shuffle) {
        std::random_device rd{};
        std::mt19937 g(rd());
        std::shuffle(m_indices.begin(), m_indices.end(), g);
    }
}

DataLoader::Iterator::Iterator(DataLoader* dataloader, int index)
    : m_dataloader(dataloader), m_index(index)
{
}

// Returns vector of (label, image) in current batch
std::vector<std::pair<int, std::shared_ptr<Tensor>>> DataLoader::Iterator::operator*() {
    std::vector<std::pair<int, std::shared_ptr<Tensor>>> batch{};
    for (int i{0}; i < m_dataloader->m_batch_size; ++i) {
        batch.push_back(m_dataloader->m_dataset->get_item(m_dataloader->m_indices[m_index + i]));
    }
    return batch;
}

std::size_t DataLoader::n_batches() const {
    return (n_samples() + batch_size() - 1) / batch_size();
}

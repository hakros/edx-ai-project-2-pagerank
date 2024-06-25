import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    allPages = set()

    linkedPages = corpus[page] if page in corpus and len(corpus[page]) > 0 else allPages

    for index in corpus:
        allPages.add(index)

    linkedPageProbability = damping_factor
    allPagesProbability = 1 - damping_factor

    # Selection of pages to choose from
    selection: set = random.choices(
        population=[
            linkedPages,
            allPages
        ],

        weights=(
            linkedPageProbability,
            allPagesProbability
        )
    )[0]

    probabilityPerPage = linkedPageProbability / len(selection)
    teleportationProbability = allPagesProbability / (len(selection) + 1)

    probabilityDict: dict = {}
    probabilityDict[page] = teleportationProbability

    for selectedPage in selection:
        probabilityDict[selectedPage] = probabilityPerPage + teleportationProbability

    return probabilityDict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    sampleLen = n
    pageRank = {}
    pages = []

    for index in corpus:
        pages.append(index)

    page = random.choice(pages)

    while n > 0:
        transitionChain = transition_model(
            corpus=corpus,
            page=page,
            damping_factor=damping_factor
        )

        choices = []
        weights = []

        for link in transitionChain:
            choices.append(link)
            weights.append(transitionChain[link])

        sample = random.choices(
            population=choices,
            weights=weights
        ).pop()

        if sample not in pageRank:
            pageRank[sample] = 0

        pageRank[sample] += 1

        page = sample

        n -= 1

    for page in pageRank:
        rank = pageRank[page]
        rank = rank / sampleLen

        pageRank[page] = rank

    return pageRank


def getAllPages(corpus):
    pages = set()

    for page in corpus:
        pages.add(page)

    return pages


def calculate_pagerank(page, numPage, corpus, pageRanks, damping_factor):
    """
    Accepts number of pages and a dictionary of links and each of their respective ranks
    """
    rank = (1 - damping_factor) / numPage

    sigmaValue = 0

    # Find all links that points to the current page
    # This is opposed to finding all pages that the current page points to
    for link in corpus:
        if len(corpus[link]) == 0:
            corpus[link] = getAllPages(corpus)

        if page not in corpus[link]:
            continue

        linkRank = pageRanks[link] / len(corpus[link])
        sigmaValue += linkRank

    rank = rank + (damping_factor * sigmaValue)

    return rank


def hasConverged(newRanks, oldRanks):
    for page in newRanks:
        # When checking for convergence we want to check in both positive and negative direction
        if abs(newRanks[page] - oldRanks[page]) > 0.001:
            return False

    return True


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    numPage = len(corpus)
    startingRank = 1 / numPage
    pageRanks = {}
    newRanks = {}
    pages = []

    print(corpus)

    for page in corpus:
        pageRanks[page] = startingRank
        pages.append(page)

    # Since newRanks is empty at first, we need to do this to prevent the while loop from stopping prematurely
    while True:
        for page in pageRanks:
            newRanks[page] = calculate_pagerank(
                page=page,
                numPage=numPage,
                pageRanks=pageRanks,
                damping_factor=damping_factor,
                corpus=corpus
            )

        if hasConverged(newRanks, pageRanks):
            break

        pageRanks = newRanks.copy()

    return pageRanks


if __name__ == "__main__":
    main()

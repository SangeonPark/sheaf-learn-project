import numpy as np
import networkx as nx

class toy_sheaf_generator(object):
    """Generate sheaves on graph with various input properties
        Nv: Number of vertices
        dv: vertex stalk dimension (Fixed to be constant for now)
        (Could also consider variable vertex stalk dimensions)
        de: edge stalk dimension (Fixed to be constant for now)
        (Could also consider variable edge stalk dimensions)
    """



    def __init__(self, Nv, dv, de):
        super(toy_sheaf_generator, self).__init__()
        self.Nv = Nv
        self.dv = dv
        self.de = de
        #self.Ne_total = Nv * (Nv-1) / 2.
        self.Pe = 1.1 * np.log(Nv) / Nv



    def generate_er_graph(self):

        #seed = np.random
        er_graph = nx.erdos_renyi_graph(self.Nv, self.Pe, seed=None)
        self.Ne = er_graph.number_of_edges()
        self.er_graph = er_graph
        return er_graph



    def random_gaussian_sheaf_on_graph(self):



        coboundary = np.zeros((self.Ne*self.de, self.Nv*self.dv))
        for i, (v1, v2) in enumerate(self.er_graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * np.random.normal(0, 1, size=(self.de, self.dv))
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* np.random.normal(0, 1, size=(self.de, self.dv))



        sheaf_laplacian = np.matmul(coboundary.T, coboundary)
        self.sheaf = sheaf_laplacian

        return sheaf_laplacian


    def sample_signal_on_sheaf(self, nsamples, smoothness=10, rng=None):
        # Sample smooth vector with tikhonov regularization

        sheaf_laplacian = self.sheaf

        n, m = sheaf_laplacian.shape

        norm = np.linalg.norm(sheaf_laplacian)

        sheaf_laplacian = sheaf_laplacian / norm

        eigenvals, eigenvecs = np.linalg.eig(sheaf_laplacian)
        smoother = np.diag(1./(1.+smoothness* eigenvals))

        #smoother = diagm((ones(n)+smoothness*lambda).^-1)
        samples = np.random.normal(0, 1, size = (n, nsamples))

        #print(eigenvals.shape, eigenvecs.shape, smoother.shape, samples.shape)

        samples_transform = eigenvecs @ smoother @ eigenvecs.T @ samples

        column_sum = samples_transform.sum(axis=0)
        samples_transform = samples_transform / column_sum[np.newaxis, :]

        return samples_transform

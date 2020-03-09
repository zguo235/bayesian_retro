import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def draw_mols_smi(smi_list, molsPerRow=5, subImgSize=(200, 200), skip_none=False, legends=None,
                  highlightAtomLists=None, highlightBondLists=None, use_default_setting=True, **kwargs):

    mols = [Chem.MolFromSmiles(x) for x in smi_list]
    if skip_none:
        idx = [i for i, mol in enumerate(mols) if mol is not None]
        mols = [mols[i] for i in idx]
        if legends is not None:
            legends = [legends[i] for i in idx]
        if highlightAtomLists is not None:
            highlightAtomLists = [highlightAtomLists[i] for i in idx]
        if highlightBondLists is not None:
            highlightBondLists = [highlightBondLists[i] for i in idx]
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow:
        nRows += 1

    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DSVG(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
    if use_default_setting:
        d2d.SetFontSize(1.3 * d2d.FontSize())
        option = d2d.drawOptions()
        # option.circleAtoms = False
        # option.continuousHighlight = False
        # option.legendFontSize = 20
        # option.multipleBondOffset = 0.07
        if legends is not None:
            option.padding = 0.11
    if legends is None:
        legends = [''] * len(mols)
    tms = [rdMolDraw2D.PrepareMolForDrawing(m) for m in mols]
    d2d.DrawMolecules(tms, legends=legends, highlightAtoms=highlightAtomLists,
                      highlightBonds=highlightBondLists, **kwargs)
    d2d.FinishDrawing()
    res = d2d.GetDrawingText()
    return res


def _okToKekulizeMol(mol, kekulize):
    if kekulize:
        for bond in mol.GetBonds():
            if bond.GetIsAromatic() and bond.HasQuery():
                return False
        return True
    return kekulize


def draw_mol_smi(smi, size=(300, 300), highlights=None, legend=None, kekulize=True,
                 use_default_setting=True, **kwargs):

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError('SMILES Parse Error')
    try:
        mol.GetAtomWithIdx(0).GetExplicitValence()
    except RuntimeError:
        mol.UpdatePropertyCache(False)

    kekulize = _okToKekulizeMol(mol, kekulize)
    d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    if use_default_setting:
        d2d.SetFontSize(1.3 * d2d.FontSize())
        option = d2d.drawOptions()
        # option.circleAtoms = False
        # option.continuousHighlight = False
        # option.legendFontSize = 20
        option.multipleBondOffset = 0.07
        if legend is not None:
            option.padding = 0.11
    try:
        mc = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize)
    except ValueError:  # <- can happen on a kekulization failure
        mc = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    # d2d.DrawMolecule(mc, legend=legend, highlightAtoms=highlights)
    d2d.DrawMolecule(mc, highlightAtoms=highlights)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg


def ReactionToImage(rxn, subImgSize=(200, 200), **kwargs):

    width = subImgSize[0] * (rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates() + 1)
    d = rdMolDraw2D.MolDraw2DSVG(width, subImgSize[1])
    d.DrawReaction(rxn, **kwargs)
    d.FinishDrawing()
    return d.GetDrawingText()


def clustering_fig(reactants_candidates_df, fig_title, fig_path):
    df = reactants_candidates_df
    grouped = df.groupby('labels')
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    # cluster_weight = grouped['weights'].mean()
    # cluster_weight.plot('bar', ax=axes[0], title="Cluster weight")
    df.boxplot(column='weights', by='labels', ax=axes[0], sym='.')
    axes[0].set_title('Cluster weight')
    cluster_size = grouped['reactants'].count()
    cluster_size.plot(kind='bar', ax=axes[1], title="Cluster size")
    fig.suptitle(fig_title)
    fig.savefig(fig_path)
    plt.close(fig)


# from liftover import get_lifter
# import mygene

# FORWARD_LIFTOVER = get_lifter("hg19", "hg38")
_GENOME_ORDER = {str(x): i for i, x in enumerate(range(1, 23), start=1)}
_GENOME_ORDER.update({"X": 23, "Y": 24, "MT": 25})

# _gene_service = mygene.MyGeneInfo()


def overlap(range1: range, range2: range):
    """Test if two ranges intersect."""
    if range1.start < range2.stop and range2.start < range1.stop:
        return True
    return False


def iterize(item):
    if isinstance(item, (str, int, float)):
        return [item]
    return item


# def convert_geneid_to_symbol(geneid, contig=None, position=None, interactive=False):
#     hit = _get_gene_info(geneid, contig, position, interactive)
#     return _convert_gene_info(hit, "symbol")
#
#
# def convert_symbol_to_geneid(symbol, contig=None, position=None, interactive=False):
#     hit = _get_gene_info(symbol, contig, position, interactive)
#     return _convert_gene_info(hit, "ensembl")


# def _get_gene_info(query, contig=None, position=None, interactive=False):
#     lookup = _gene_service.query(
#         query,
#         fields=["ensembl.gene", "genomic_pos", "symbol"],
#         species="human"
#         )
#     if lookup["total"] == 1:
#         return lookup["hits"][0]
#     if lookup["total"] == 0:
#         raise Warning(f"Gene lookup for '{query}' had 0 hits.")
#     elif lookup["total"] > 1 and contig is None or position is None:
#         raise Warning(f"Gene lookup for '{query}' had {lookup['total']} results. "
#                       f"Provide contig and position to attempt resolution:\n"
#                       f"{lookup['hits']}")
#     subhits = []
#     for hit in lookup["hits"]:
#         if "genomic_pos" not in hit:
#             continue
#         pos = hit["genomic_pos"]
#         if isinstance(pos, list):
#             pos = pos[0]
#         if pos["chr"] == str(contig) and pos["start"] <= position <= pos["end"]:
#             subhits.append(hit)
#     if len(subhits) == 1:
#         return subhits[0]
#     if interactive:
#         ids = [(hit["ensembl"]["gene"], hit["symbol"]) for hit in subhits]
#         ids = {x for result in ids for x in result}
#         if query in ids and len(subhits) <= 5:
#             print(f"Multiple possible matches found for '{query}'. Select match:")
#             print("\tN: None of these (cancel and exit)")
#             print(*[f"\t{i}: {x}\n" for i, x in enumerate(subhits, start=0)])
#             selection = input()
#             if selection not in "01234":
#                 print("Exitting.")
#                 exit()
#             return subhits[int(selection)]
#     raise Warning(f"Gene lookup for '{query}' had {len(subhits)} possible subhits. "
#               f"Could not resolve:"
#               f"{subhits}")


def _convert_gene_info(info_hit, get):
    if get == "symbol":
        return info_hit["symbol"]
    elif get == "ensembl":
        return info_hit["ensembl"]["gene"]
    else:
        raise ValueError(f"Unknown request to convert to '{get}'.")

# def get_exon_overlap(position, exon_fields):
#
#     for i, (start, stop) in enumerate(exons["positions"]):
#         if (start <= position <= stop) or (stop < position < exons[i][0]):
#             return i
#         if position > stop
#!/usr/bin/env python3
"""
Script to create a sybil attack scenario by duplicating all validators
from top_tier.json and prefixing their IDs and names with "Sybil ".
The sybil validators will have identical quorum set structures but reference
only other sybil validators, creating two disjoint networks.
"""

import json
from python_fbas.fbas_graph import FBASGraph
from python_fbas.serialization import FBASSerializer


def generate_sybil_id(original_id: str) -> str:
    """Generate a sybil validator ID by prefixing with 'Sybil '"""
    return f"Sybil {original_id}"


def create_sybil_validator_copy(original_validator: dict) -> dict:
    """Create a sybil copy of a validator with modified ID and name"""
    sybil = original_validator.copy()
    sybil["id"] = generate_sybil_id(original_validator["id"])

    # Create a sybil qset reference
    original_qset = original_validator.get("qset", "_q0")
    sybil["qset"] = f"{original_qset}_sybil"

    # Update the name and homeDomain in attrs
    if "attrs" in sybil and "name" in sybil["attrs"]:
        sybil["attrs"] = sybil["attrs"].copy()
        sybil["attrs"]["name"] = f"Sybil {sybil['attrs']['name']}"
        sybil["attrs"]["homeDomain"] = f"Sybil {sybil['attrs']['homeDomain']}"

    return sybil


def create_sybil_qset_id(original_qset_id: str) -> str:
    """Create a sybil qset ID"""
    return f"{original_qset_id}_sybil"


def duplicate_qset_with_sybils(
        original_qset: dict,
        validator_mapping: dict) -> dict:
    """Create a qset that references sybil validators/qsets instead of originals"""
    sybil_qset = {
        "threshold": original_qset["threshold"],
        "members": []
    }

    for member in original_qset["members"]:
        if member in validator_mapping:
            # This is a validator ID, replace with sybil version
            sybil_qset["members"].append(validator_mapping[member])
        elif member.startswith("_q"):
            # This is a qset reference, use the sybil version
            sybil_qset["members"].append(create_sybil_qset_id(member))
        else:
            # Keep as is (shouldn't happen in normal cases)
            sybil_qset["members"].append(member)

    return sybil_qset


def create_sybil_fbas_variant(input_file: str, output_file: str, traitor_homedomains: list[str], traitor_only_sybil: bool = False):
    """
    Create a sybil FBAS variant using the FBASGraph API.

    Args:
        input_file: Input JSON file
        output_file: Output JSON file
        traitor_homedomains: List of homedomains that will be traitors
        traitor_only_sybil: If True, traitors only reference sybil qsets (threshold 5).
                           If False, traitors reference both original and sybil qsets (threshold 10).
    """

    print(f"Loading FBAS from {input_file}...")
    with open(input_file, 'r') as f:
        serializer = FBASSerializer()
        original_fbas = serializer.deserialize(f.read())

    print(f"Original FBAS has {len(original_fbas.get_validators())} validators")
    print(f"Traitor homedomains: {traitor_homedomains}")

    # Create new FBAS that will contain original + sybil + traitor structure
    fbas = FBASGraph()

    # Add all original validators with their attributes
    original_validators = list(original_fbas.get_validators())
    for validator_id in original_validators:
        fbas.add_validator(validator_id)
        # Copy attributes
        attrs = original_fbas.vertice_attrs(validator_id)
        fbas.update_validator(validator_id, None, **attrs)

    # Create mapping from original validator IDs to sybil IDs
    validator_mapping = {}
    for validator_id in original_validators:
        validator_mapping[validator_id] = generate_sybil_id(validator_id)

    # Add sybil validators
    sybil_validators = []
    for validator_id in original_validators:
        sybil_id = generate_sybil_id(validator_id)
        fbas.add_validator(sybil_id)
        sybil_validators.append(sybil_id)

        # Copy and modify attributes
        original_attrs = original_fbas.vertice_attrs(validator_id)
        sybil_attrs = original_attrs.copy()
        if "name" in sybil_attrs:
            sybil_attrs["name"] = f"Sybil {sybil_attrs['name']}"
        if "homeDomain" in sybil_attrs:
            sybil_attrs["homeDomain"] = f"Sybil {sybil_attrs['homeDomain']}"
        fbas.update_validator(sybil_id, None, **sybil_attrs)

    print(f"Created {len(sybil_validators)} sybil validators")

    # Recreate qset structure using the API
    qset_mapping = {}  # Maps original qset IDs to new qset IDs
    sybil_qset_mapping = {}  # Maps original qset IDs to sybil qset IDs

    # Get all qset nodes from original graph, process them in dependency order
    original_qset_nodes = [q for q in original_fbas.vertices() if not original_fbas.is_validator(q)]

    # Create original qsets first (maintaining structure)
    def create_qsets_recursive(qset_id, is_sybil=False):
        if is_sybil:
            if qset_id in sybil_qset_mapping:
                return sybil_qset_mapping[qset_id]
            target_mapping = sybil_qset_mapping
            id_suffix = "_sybil"
        else:
            if qset_id in qset_mapping:
                return qset_mapping[qset_id]
            target_mapping = qset_mapping
            id_suffix = ""

        threshold = original_fbas.threshold(qset_id)
        original_members = list(original_fbas.graph_view().successors(qset_id))

        # Convert members to new graph context
        new_members = []
        for member in original_members:
            if original_fbas.is_validator(member):
                # It's a validator
                if is_sybil:
                    new_members.append(validator_mapping[member])
                else:
                    new_members.append(member)
            else:
                # It's a qset - recursively create it
                new_qset_id = create_qsets_recursive(member, is_sybil)
                new_members.append(new_qset_id)

        # Create the qset using the API
        suggested_id = f"{qset_id}{id_suffix}" if qset_id != "_q0" else f"_q0{id_suffix}"
        new_qset_id = fbas.add_qset(threshold=threshold, members=new_members, qset_id=suggested_id)
        target_mapping[qset_id] = new_qset_id
        return new_qset_id

    # Create all original qsets
    for qset_id in original_qset_nodes:
        create_qsets_recursive(qset_id, is_sybil=False)

    # Create all sybil qsets
    for qset_id in original_qset_nodes:
        create_qsets_recursive(qset_id, is_sybil=True)

    print(f"Created {len(qset_mapping)} original qsets and {len(sybil_qset_mapping)} sybil qsets")

    # Create the traitor qset using the API
    if traitor_only_sybil:
        # Reference sybil second-level qsets + traitor qsets with threshold 7
        traitor_qset_members = []
        for original_qset_id in original_qset_nodes:
            if original_qset_id != "_q0":
                traitor_qset_members.append(sybil_qset_mapping[original_qset_id])  # sybil versions

        # Add traitor qsets for the traitor homedomains
        traitor_inner_qsets = []
        for i, domain in enumerate(traitor_homedomains):
            # Find validators from this domain
            domain_validators = []
            for validator_id in original_validators:
                validator_domain = original_fbas.vertice_attrs(validator_id).get("homeDomain", "")
                if validator_domain == domain:
                    domain_validators.append(validator_id)

            if domain_validators:
                # Create inner qset with majority threshold for domain validators
                threshold = len(domain_validators) // 2 + 1  # majority
                traitor_inner_qset_id = fbas.add_qset(
                    threshold=threshold,
                    members=domain_validators,
                    qset_id=f"_qtraitor_{i+1}"
                )
                traitor_inner_qsets.append(traitor_inner_qset_id)
                traitor_qset_members.append(traitor_inner_qset_id)

        traitor_qset_id = fbas.add_qset(
            threshold=7,  # threshold of 7 out of 9 (7 sybil + 2 traitor qsets)
            members=traitor_qset_members,
            qset_id="_q0_traitor_sybil"
        )
        print(f"Created traitor-sybil qset with threshold 7 and {len(traitor_qset_members)} members (7 sybil + {len(traitor_inner_qsets)} traitor qsets)")
    else:
        # Reference both original and sybil second-level qsets with threshold 10
        traitor_qset_members = []
        for original_qset_id in original_qset_nodes:
            if original_qset_id != "_q0":
                traitor_qset_members.append(qset_mapping[original_qset_id])  # original
                traitor_qset_members.append(sybil_qset_mapping[original_qset_id])  # sybil version

        traitor_qset_id = fbas.add_qset(
            threshold=10,  # threshold of 10 out of 14 second-level qsets
            members=traitor_qset_members,
            qset_id="_q0_traitor"
        )
        print(f"Created traitor qset with threshold 10 and {len(traitor_qset_members)} members")

    # Connect validators to their qsets and identify traitors
    traitor_validators = []

    # Connect original validators to their original qsets
    for validator_id in original_validators:
        if original_fbas.graph_view().out_degree(validator_id) == 1:
            original_qset_id = original_fbas.qset_vertex_of(validator_id)
            new_qset_id = qset_mapping[original_qset_id]

            # Check if this validator should be a traitor
            validator_domain = original_fbas.vertice_attrs(validator_id).get("homeDomain", "")
            if validator_domain in traitor_homedomains:
                fbas.update_validator(validator_id, traitor_qset_id)
                traitor_validators.append(validator_id)
            else:
                fbas.update_validator(validator_id, new_qset_id)

    # Connect sybil validators to their sybil qsets
    for i, validator_id in enumerate(original_validators):
        sybil_id = sybil_validators[i]
        if original_fbas.graph_view().out_degree(validator_id) == 1:
            original_qset_id = original_fbas.qset_vertex_of(validator_id)
            sybil_qset_id = sybil_qset_mapping[original_qset_id]

            # Check if this sybil validator should be a traitor
            validator_domain = original_fbas.vertice_attrs(validator_id).get("homeDomain", "")
            if validator_domain in traitor_homedomains:
                fbas.update_validator(sybil_id, traitor_qset_id)
                traitor_validators.append(sybil_id)
            else:
                fbas.update_validator(sybil_id, sybil_qset_id)

    print(f"Updated {len(traitor_validators)} validators to use traitor qset")

    # Get traitor validator names for display
    traitor_names = []
    for validator_id in traitor_validators:
        name = fbas.vertice_attrs(validator_id).get("name", validator_id)
        traitor_names.append(name)

    print(f"Traitor validators: {traitor_names}")

    # Get stats
    qset_nodes = [q for q in fbas.vertices() if not fbas.is_validator(q)]
    total_validators = len(fbas.get_validators())
    total_qsets = len(qset_nodes)

    print(f"Final FBAS has {total_validators} validators and {total_qsets} qsets")

    # Save result using the API
    print(f"Saving to {output_file}...")
    serializer = FBASSerializer()
    json_str = serializer.serialize(fbas, format='python-fbas')
    with open(output_file, 'w') as f:
        f.write(json_str)

    print("Done!")
    print(f"Original validators: {len(original_validators)}")
    print(f"Sybil validators: {len(sybil_validators)}")
    print(f"Traitor validators: {len(traitor_validators)} (from {len(traitor_homedomains)} homedomains)")
    print(f"Total validators: {total_validators}")
    print(f"Original qsets: {len(qset_mapping)}")
    print(f"Sybil qsets: {len(sybil_qset_mapping)}")
    if traitor_only_sybil:
        print(f"Traitor qsets: 1 top-level + {len(traitor_inner_qsets) if 'traitor_inner_qsets' in locals() else 0} inner qsets")
        print("Traitor qset references SYBIL second-level qsets + traitor inner qsets!")
        print("This creates a scenario where traitors are aligned with the sybil network!")
    else:
        print(f"Traitor qsets: 1")
        print("Traitor qset references both ORIGINAL and SYBIL second-level qsets!")
        print("This creates a scenario with sybil networks and traitor validators!")
    print(f"Total qsets: {total_qsets}")
    print(f"Traitor homedomains: {traitor_homedomains}")

    # Check that the new FBAS passes integrity check
    fbas.check_integrity()
    print(f"Successfully created FBAS with {len(fbas.get_validators())} validators!")

    return fbas


def main():
    """Create both sybil_1.json and sybil_2.json variants"""
    input_file = "top_tier.json"

    # Select 2 homedomains for traitor behavior
    traitor_homedomains = [
        "lobstr.co",
        "satoshipay.io"
    ]

    print("=" * 80)
    print("CREATING SYBIL_1.JSON - Traitors reference both original and sybil qsets")
    print("=" * 80)

    # Create sybil_1.json: traitors reference both original and sybil qsets (threshold 10)
    create_sybil_fbas_variant(
        input_file=input_file,
        output_file="sybil_1.json",
        traitor_homedomains=traitor_homedomains,
        traitor_only_sybil=False
    )

    print("\n" + "=" * 80)
    print("CREATING SYBIL_2.JSON - Traitors reference only sybil qsets")
    print("=" * 80)

    # Create sybil_2.json: traitors reference only sybil qsets (threshold 5)
    create_sybil_fbas_variant(
        input_file=input_file,
        output_file="sybil_2.json",
        traitor_homedomains=traitor_homedomains,
        traitor_only_sybil=True
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Created two sybil attack scenarios:")
    print("- sybil_1.json: Traitors can bridge original and sybil networks (threshold 10/14)")
    print("- sybil_2.json: Traitors are aligned with sybil network + traitor domains (threshold 7/9)")
    print(f"- Both use traitor homedomains: {traitor_homedomains}")


if __name__ == "__main__":
    main()

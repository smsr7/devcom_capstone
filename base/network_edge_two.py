class NetworkEdge:
    def __init__(self, data):
        """
        Constructor for the NetworkEdge object

        Parameters:
            data - The JSON edge object, expected to include:
                'from', 'to', 'landmine_present',
                'human_estimate', 'ai_estimate',
                'high_ai_estimate', 'metadata'
        """
        self.origin            = data['from']
        self.destination       = data['to']
        self.landmine_present  = data['landmine_present']
        self.landmine_found    = False
        self.landmine_cleared  = False

        # primary estimates
        self.human_estimate    = data['human_estimate']
        self.ai_estimate       = data['ai_estimate']

        # high-reliability AI estimate for the second UAV
        self.high_ai_estimate  = data.get('high_ai_estimate', None)

        # environmental metadata (terrain, time, etc.)
        self.metadata          = data['metadata']

        # query flags
        self.uav_scanned           = False  # primary UAV has scanned
        self.ai_queried            = False  # primary AI queried
        self.human_queried         = False  # human queried

        self.scanner_uav_scanned   = False  # second UAV has scanned
        self.high_ai_queried       = False  # high‐AI queried by second UAV

    def __eq__(self, other) -> bool:
        """
        Two edges are equal if they connect the same nodes (undirected).
        """
        if isinstance(other, NetworkEdge):
            return ((self.origin == other.origin and self.destination == other.destination) or
                    (self.origin == other.destination and self.destination == other.origin))
        return False

    def __hash__(self) -> int:
        """
        Include high_ai_estimate so edges with different high‐AI predictions
        hash differently.
        """
        return hash((
            self.origin,
            self.destination,
            self.human_estimate,
            self.ai_estimate,
            self.high_ai_estimate
        ))
